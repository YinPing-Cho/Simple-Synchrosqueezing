import librosa
import torch
import numpy as np
import scipy.signal as sig
from scipy.fft import ifft
import soundfile as sf
import matplotlib.pyplot as plt
import time
from utils import calc_SNR, pre_emphasis, zero_padding, make_frames, torch_MAD, find_closest, indexed_sum, de_zero_pad

class Synchrosqueezing:
    def __init__(self, torch_device):
        self.torch_device = torch_device
        torch.set_grad_enabled(False)

        self.pi = np.pi
        self.EPS32 = np.finfo(np.float32).eps
        self.EPS64 = np.finfo(np.float64).eps
        self.signal_length = None
        self.zero_padded_length = None
        self.sr = None
        self.time_run = None
        self.window = None
        self.win_length = None
        self.hop_length = None
        self.n_fft = None

    def xi_function(self, scale, N):
        """
        Credit: https://github.com/OverLordGoldDragon/ssqueezepy/blob/efc6f916974be6459f92b35fb21d17b3d7553ed5/ssqueezepy/wavelets.py#L473
        N=128: [0, 1, 2, ..., 64, -63, -62, ..., -1] * (2*pi / N) * scale
        N=129: [0, 1, 2, ..., 64, -64, -63, ..., -1] * (2*pi / N) * scale
        """
        xi = torch.zeros(N, device=torch_device)
        h = scale * (2 * self.pi) / N
        for i in range(N // 2 + 1):
            xi[i] = i * h
        for i in range(N // 2 + 1, N):
            xi[i] = (i - N) * h
        return xi

    def win_and_diff_win(self, win_length, n_fft):
        #window = torch.tensor(sig.windows.dpss(win_length, max(4, win_length//8), sym=False), device=self.torch_device)
        window = torch.hann_window(win_length, device=self.torch_device)
        window = zero_padding(window, n_fft)
        #plt.plot(window.detach().cpu().numpy())
        #plt.show()
        
        ffted_win = torch.fft.fft(window)
        n_win = len(window)
        xi = self.xi_function(1, n_win)
        if n_win % 2 == 0:
            xi[n_win // 2] = 0
        diffed_win = torch.fft.ifft(ffted_win * 1j * xi).real

        return window, diffed_win

    def stft_dstft(self, audio, win_length, hop_length, n_fft):
        '''
        1) Obtain window and "differenced" window (passed into tensor)
        2) Zero-pad audio according to the n_fft length
        3) slice audio into frames and multiply it with both window and diffed_window
        4) FFT the windowed frames to have the STFTed audio as Sx and differenced-STFTed audio as dSx
        '''
        window, diffed_window = self.win_and_diff_win(win_length, n_fft)
        self.window = window

        audio = torch.nn.functional.pad(audio, (0, n_fft))
        framed_audio = make_frames(audio, n_fft, hop_length)

        Sx = framed_audio * window.reshape((-1, 1))
        dSx = framed_audio * diffed_window.reshape((-1, 1))

        '''
        Dimensions: n_fft*num_frames -> (b_fft//2+1)*num_frames
        '''
        ffted_Sx = torch.fft.rfft(Sx, dim=0)
        ffted_dSx = torch.fft.rfft(dSx, dim=0)*self.sr

        return ffted_Sx, ffted_dSx

    def get_Sfs(self, Sx):
        Sfs = torch.linspace(0, .5*self.sr, Sx.size(0), device=Sx.device)
        return Sfs

    def phase_transform(self, Sx, dSx, Sfs, gamma):
        '''
        STFT phase transform: w[u, k] = Im( k - d/dt(Sx[u, k]) / Sx[u, k] / (j*2pi) )
        which can be reduced to the form here as explained in: https://dsp.stackexchange.com/a/72589/50076
        '''
        w = Sfs.reshape(-1, 1) - torch.imag(dSx / Sx) / (2*self.pi)
        w = torch.abs(w)

        if gamma is None:
            gamma = np.sqrt(self.EPS32)
        elif gamma[0] == 'adaptive':
            gamma = torch_MAD(Sx) * 1.4826 * np.sqrt(2*np.log(self.signal_length)) * gamma[1]
        else:
            raise ValueError('gamma option {} not implemented; support `None` or `(`adaptive`, float)`')

        '''
        mark noise as inf to be later removed
        '''
        w[torch.abs(Sx) < gamma] = np.inf
        return w
    
    def replace_inf(self, x, inf_criterion):
        x = torch.where(torch.isinf(inf_criterion), torch.zeros_like(x), x)
        return x

    def synchrosqueeze(self, Sx, w, ssq_freqs, squeezetype='abs'):
        assert not (torch.min(w) < 0), 'max: {}; min: {}'.format(torch.max(w), torch.min(w))

        #########################################
        '''
        Amplitude manipulation on Sx
        '''
        if squeezetype == 'lebesque':
            Sx = torch.ones_like(Sx) / Sx.size(0)
        elif squeezetype == 'abs':
            Sx = torch.abs(Sx)
        else:
            raise ValueError('Unsupported squeeze function keyword; support `lebesque` or `abs`.')
        #########################################

        #########################################
        '''
        Remove noise data points which were previously marked as inf in w
        '''
        if self.time_run:
            torch.cuda.synchronize()
            start_time = time.time()
        Sx = self.replace_inf(Sx, inf_criterion=w)
        if self.time_run:
            torch.cuda.synchronize()
            print("Clear inf time: %s seconds ---" % (time.time() - start_time))
        #########################################

        #########################################
        '''
        Squeeze by spectral adjacency
        '''
        if self.time_run:
            torch.cuda.synchronize()
            start_time = time.time()
        freq_mod_indices = find_closest(w.contiguous(), ssq_freqs.contiguous())
        if self.time_run:
            torch.cuda.synchronize()
            print("Spectral squeezing time: %s seconds ---" % (time.time() - start_time))
        #########################################

        #########################################
        '''
        T-F reassignment according to the squeezed result
        '''
        if self.time_run:
            torch.cuda.synchronize()
            start_time = time.time()
        df = ssq_freqs[1] - ssq_freqs[0]
        Tx = indexed_sum(Sx * df, freq_mod_indices)
        if self.time_run:
            torch.cuda.synchronize()
            print("Indexed sum time: %s seconds ---" % (time.time() - start_time))
        #########################################

        return Tx
    
    def audio_preparation(self, audio):
        '''
        1) pad the audio to proper length such that FFT can be done efficiently
        2) dt is calculated as the difference (integrand) under the sample rate
        3) convert everything to Pytorch tensors
        '''
        audio = torch.tensor(audio, device=torch_device)
        time_indices = torch.linspace(start=0,end=dur,steps=audio.size(0), device=torch_device)

        dt = time_indices[1]-time_indices[0]
        assert torch.isclose(dt,dur/torch.tensor(audio.size(0)),atol=1e-5), '{} and {}'.format(dt, dur/audio.size(0))
        audio = zero_padding(audio)
        self.zero_padded_length = audio.size(0)

        return audio, dt

    def visualize(self, T):
        T = np.log(np.abs(T)+1e-3)
        T = T - np.min(T)
        plt.imshow(np.flipud(T), aspect='auto', cmap='jet') # , vmin=0, vmax=.2
        plt.show()

    def sst_stft_forward(self, audio, sr, gamma=None, win_length=512, hop_length=128, n_fft=512, visualize=True, time_run=True):
        self.signal_length = int((audio.shape[0]//2)*2)
        self.time_run = time_run
        self.sr = sr
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft

        if self.time_run:
            sst_start_time = time.time()

        #########################################
        '''
        Audio Preparation
        '''
        audio, dt = self.audio_preparation(audio)
        #########################################

        #########################################
        if self.time_run:
            torch.cuda.synchronize()
            start_time = time.time()
        '''
        Obtain STFTed signal and STFTed "difference" signal
        '''
        Sx, dSx = self.stft_dstft(audio, win_length, hop_length, n_fft)
        Sfs = self.get_Sfs(Sx)
        if self.time_run:
            torch.cuda.synchronize()
            print("Sx and dSx stft time: %s seconds ---" % (time.time() - start_time))
        #########################################

        #########################################
        if self.time_run:
            torch.cuda.synchronize()
            start_time = time.time()
        '''
        Phase transform
        '''
        w = self.phase_transform(Sx, dSx, Sfs, gamma)
        if self.time_run:
            torch.cuda.synchronize()
            print("Phase transform time: %s seconds ---" % (time.time() - start_time))
        #########################################

        #########################################
        '''
        Synchrosqueeze on the STFT
        '''
        # Tx is returned as numpy array
        Tx = self.synchrosqueeze(Sx, w, ssq_freqs=Sfs, squeezetype='lebesque')
        Sx = Sx.detach().cpu().numpy()
        #########################################
        if self.time_run:
            print("--- SST total run time: %s seconds ---" % (time.time() - sst_start_time))

        Tx = de_zero_pad(Tx, self.signal_length, self.zero_padded_length, hop_length)
        Sx = de_zero_pad(Sx, self.signal_length, self.zero_padded_length, hop_length)

        if visualize:
            self.visualize(Tx)
            self.visualize(Sx)
        return Tx, Sx
    
    def sst_stft_inverse(self, Tx):
        signal = np.zeros(self.signal_length+self.win_length)
        spectral_channels = Tx.shape[0]
        window = torch.hann_window(spectral_channels, device='cpu')
        window = torch.nn.functional.pad(window, (0, self.win_length-spectral_channels))
        window = window.numpy()

        for idx in range(self.signal_length//self.hop_length):
            frame = np.zeros(self.win_length)
            frame[:spectral_channels] += ifft(Tx[:, idx]).real
            frame *= window
            signal[idx*self.hop_length:idx*self.hop_length+self.win_length] += frame

        return signal

torch_device = 'cuda'
filename = 'draw_16.wav'
audio, sr = sf.read(filename, dtype='float32')
dur = librosa.get_duration(y=audio, sr=sr)
print('Audio name: {};\nSample rate: {};\nDuration: {};\nShape: {};'.format(filename, sr, dur, audio.shape))
#audio = pre_emphasis(audio)

SST = Synchrosqueezing(torch_device=torch_device)

N = 8192
NyqF = N/2
time_len = 10
t = np.linspace(0, time_len, N*time_len, endpoint=False)
xo = np.cos(2*np.pi*np.sin(t)*np.cos(t/time_len)*NyqF)
xo += np.cos(2*np.pi*np.cos(t)*np.cos(t/time_len*2)*NyqF)
xo += np.cos(2*np.pi*np.cos(t)*np.cos(t/time_len*5)*NyqF)
xo /= np.max(np.abs(xo))
x = xo + 0.5*np.random.standard_normal(N*time_len)  # add noise

print(x.shape)

input_signal = x
sr = N

print('The SNR of input signal: {} dB'.format(calc_SNR(input_signal)))

sf.write('input.wav', data=input_signal, samplerate=sr, subtype='PCM_16')
Tx, Sx = SST.sst_stft_forward(audio=input_signal, sr=sr, gamma=('adaptive',0.9), visualize=True, time_run=True)
print('Tx max: {}; min: {}'.format(np.max(np.abs(Tx)), np.min(np.abs(Tx))))
#plt.plot(Tx)
#plt.show()

recon_signal = SST.sst_stft_inverse(Tx)
recon_signal /= np.max(np.abs(recon_signal))

print('The SNR of recon signal: {} dB'.format(calc_SNR(recon_signal)))

sf.write('recon.wav', data=recon_signal, samplerate=sr, subtype='PCM_16')