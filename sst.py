import librosa
import torch
import numpy as np
import scipy.signal as sig
import soundfile as sf
import matplotlib.pyplot as plt
import time
from utils import zero_padding, make_frames, find_closest, indexed_sum

class Synchrosqueezing:
    def __init__(self, torch_device):
        self.torch_device = torch_device
        torch.set_grad_enabled(False)

        self.pi = np.pi
        self.EPS32 = np.finfo(np.float32).eps
        self.EPS64 = np.finfo(np.float64).eps
        self.sr = None
        self.time_run = None

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
        window = torch.tensor(sig.windows.dpss(win_length, max(4, win_length//8), sym=False), device=self.torch_device)
        window = zero_padding(window, n_fft)
        
        ffted_win = torch.fft.fft(window)
        n_win = len(window)
        xi = self.xi_function(1, n_win)
        if n_win % 2 == 0:
            xi[n_win // 2] = 0
        diffed_win = torch.fft.ifft(ffted_win * 1j * xi).real

        return window, diffed_win

    def stft_dstft(self, audio, win_length, hop_length, n_fft):
        window, diffed_window = self.win_and_diff_win(win_length, n_fft)
        '''
        plt.plot(window)
        plt.show()
        plt.clf()
        plt.plot(diffed_window)
        plt.show()
        plt.clf()
        '''
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

        med = (dSx / Sx) / (2*self.pi)

        w = Sfs.reshape(-1, 1) - torch.imag(dSx / Sx) / (2*self.pi)
        w = torch.abs(w)

        if gamma is None:
            gamma = np.sqrt(self.EPS32)
        elif gamma == 'adaptive':
            gamma = torch.mean(torch.abs(Sx)) * 0.99

        w[torch.abs(Sx) < gamma] = np.inf
        return w
    
    def replace_inf(self, x, inf_criterion):
        x = torch.where(torch.isinf(inf_criterion), torch.zeros_like(x), x)
        return x

    def synchrosqueeze(self, Sx, w, ssq_freqs, squeezetype='lebesque'):
        assert not (torch.min(w) < 0), 'max: {}; min: {}'.format(torch.max(w), torch.min(w))

        if squeezetype == 'lebesque':
            Sx = torch.ones_like(Sx) / Sx.size(0)
        elif squeezetype == 'abs':
            Sx = torch.abs(Sx)
        else:
            raise ValueError('Unsupported squeeze function keyword; support `lebesque` or `abs`.')
        
        if self.time_run:
            torch.cuda.synchronize()
            start_time = time.time()
        Sx = self.replace_inf(Sx, inf_criterion=w)
        if self.time_run:
            torch.cuda.synchronize()
            print("Clear inf time: %s seconds ---" % (time.time() - start_time))

        if self.time_run:
            torch.cuda.synchronize()
            start_time = time.time()
        # squeeze by spectral adjacency
        freq_mod_indices = find_closest(w.contiguous(), ssq_freqs.contiguous())
        if self.time_run:
            torch.cuda.synchronize()
            print("Spectral squeezing time: %s seconds ---" % (time.time() - start_time))

        if self.time_run:
            torch.cuda.synchronize()
            start_time = time.time()
        df = ssq_freqs[1] - ssq_freqs[0]
        Tx = indexed_sum(Sx * df, freq_mod_indices)
        if self.time_run:
            print("Indexed sum time: %s seconds ---" % (time.time() - start_time))

        return Tx
    
    def audio_preparation(self, audio):
        audio = torch.tensor(audio, device=torch_device)
        time_indices = torch.linspace(start=0,end=dur,steps=audio.size(0), device=torch_device)

        dt = time_indices[1]-time_indices[0]
        assert torch.isclose(dt,dur/torch.tensor(audio.size(0)),atol=1e-5), '{} and {}'.format(dt, dur/audio.size(0))
        audio = zero_padding(audio)

        return audio, dt

    def visualize(self, T):
        T = T.detach().cpu()
        plt.imshow(torch.abs(T)/torch.max(torch.abs(T)), aspect='auto', vmin=0, vmax=.2, cmap='jet')
        plt.show()

    def sst_stft_forward(self, audio, sr, gamma=None, win_length=256, hop_length=1, n_fft=256, visualize=True, time_run=True):
        self.time_run = time_run
        self.sr = sr
        if self.time_run:
            sst_start_time = time.time()
        #########################################
        audio, dt = self.audio_preparation(audio)
        
        if self.time_run:
            torch.cuda.synchronize()
            start_time = time.time()
        Sx, dSx = self.stft_dstft(audio, win_length, hop_length, n_fft)
        Sfs = self.get_Sfs(Sx)
        if self.time_run:
            torch.cuda.synchronize()
            print("Sx and dSx stft time: %s seconds ---" % (time.time() - start_time))

        if self.time_run:
            torch.cuda.synchronize()
            start_time = time.time()
        w = self.phase_transform(Sx, dSx, Sfs, gamma)
        if self.time_run:
            torch.cuda.synchronize()
            print("Phase transform time: %s seconds ---" % (time.time() - start_time))
        
        Tx = self.synchrosqueeze(Sx, w, ssq_freqs=Sfs, squeezetype='lebesque')
        #########################################
        if self.time_run:
            print("--- SST total run time: %s seconds ---" % (time.time() - sst_start_time))

        if visualize:
            self.visualize(Tx)
            self.visualize(Sx)
        return Tx.detach().cpu(), Sx.detach().cpu()

torch_device = 'cuda'
filename = 'draw_16.wav'
audio, sr = sf.read(filename, dtype='float32')
dur = librosa.get_duration(y=audio, sr=sr)
print('Audio name: {};\nSample rate: {};\nDuration: {};\nShape: {};'.format(filename, sr, dur, audio.shape))

SST = Synchrosqueezing(torch_device=torch_device)

N = 2048
t = np.linspace(0, 10, N, endpoint=False)
xo = np.cos(2 * np.pi * 2 * (np.exp(t / 2.2) - 1))
xo += xo[::-1]  # add self reflected
x = xo + np.sqrt(2) * np.random.randn(N)  # add noise

print(x.shape)
Tx, Sx = SST.sst_stft_forward(audio=xo, sr=N, gamma='adaptive', visualize=True, time_run=True)
#print('Tx max: {}; min: {}'.format(torch.max(Tx.abs()), torch.min(Tx.abs())))
plt.plot(Tx)
plt.show()