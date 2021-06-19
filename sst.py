import librosa
import torch
import numpy as np
import scipy.signal as sig
#from scipy.fft import fft, ifft, rfft
from numpy.fft import fft, ifft, rfft, ifftshift
from scipy.signal import istft as istft
import matplotlib.pyplot as plt
import time
from utils import pre_emphasis, make_frames, torch_MAD, find_closest, indexed_sum

class Synchrosqueezing:
    def __init__(self, torch_device):
        self.torch_device = torch_device
        torch.set_grad_enabled(False)

        self.visualizeFigs = False
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
        '''
        Credit: https://github.com/OverLordGoldDragon/ssqueezepy/blob/efc6f916974be6459f92b35fb21d17b3d7553ed5/ssqueezepy/wavelets.py#L473
        N=128: [0, 1, 2, ..., 64, -63, -62, ..., -1] * (2*pi / N) * scale
        N=129: [0, 1, 2, ..., 64, -64, -63, ..., -1] * (2*pi / N) * scale
        '''
        xi = torch.zeros(N, device=self.torch_device)
        h = scale * (2 * self.pi) / N
        for i in range(N // 2 + 1):
            xi[i] = i * h
        for i in range(N // 2 + 1, N):
            xi[i] = (i - N) * h
        return xi

    def win_and_diff_win(self, win_length, n_fft, use_Hann=False):
        if use_Hann:
            window = torch.hann_window(win_length, device=self.torch_device)
        else:
            window = torch.tensor(sig.windows.dpss(win_length, max(4, win_length//8), sym=False), device=self.torch_device)
            
        ffted_win = torch.fft.fft(window)
        n_win = len(window)
        xi = self.xi_function(1, n_win)
        if n_win % 2 == 0:
            xi[n_win // 2] = 0
        diffed_win = torch.fft.ifft(ffted_win * 1j * xi).real

        if self.visualizeFigs:
            plt.plot(window.detach().cpu().numpy())
            plt.title('Window shape')
            plt.show()
            plt.plot(diffed_win.detach().cpu().numpy())
            plt.title('Differenced window')
            plt.show()

        return window, diffed_win

    def stft_dstft(self, audio, win_length, hop_length, n_fft, use_Hann):
        '''
        1) Obtain window and "differenced" window (passed into tensor)
        2) Zero-pad audio according to the n_fft length
        3) slice audio into frames and multiply it with both window and diffed_window
        4) FFT the windowed frames to have the STFTed audio as Sx and differenced-STFTed audio as dSx
        '''
        window, diffed_window = self.win_and_diff_win(win_length, n_fft, use_Hann=use_Hann)
        self.window = window
        
        audio = np.pad(audio, (n_fft//2, n_fft//2-1), mode='reflect')
        audio = torch.tensor(audio, dtype=torch.complex64, device=self.torch_device)
        self.zero_padded_length = audio.shape[0]

        framed_audio = make_frames(audio, n_fft, hop_length)

        Sx = framed_audio * window.reshape(-1, 1)
        dSx = framed_audio * diffed_window.reshape(-1, 1)

        Sx = torch.fft.ifftshift(Sx, dim=0).real
        dSx = torch.fft.ifftshift(dSx, dim=0).real

        if self.visualizeFigs:
            self.visualize(dSx, dBscale=True, title='Window-Differenced STFT result in dB scale')
        '''
        Dimensions: n_fft*num_frames -> (b_fft//2+1)*num_frames
        '''
        ffted_Sx = torch.fft.rfft(Sx, dim=0)
        ffted_dSx = torch.fft.rfft(dSx, dim=0)*1.0
        return ffted_Sx, ffted_dSx

    def get_Sfs(self, Sx):
        Sfs = torch.linspace(0, .5*1.0, Sx.size(0), device=self.torch_device)
        return Sfs

    def phase_transform(self, Sx, dSx, Sfs, gamma):
        '''
        STFT phase transform.
        Per: STFT-SST: Thakur, Gaurav and Wu, Hau-Tieng,
        “Synchrosqueezing-Based Recovery of Instantaneous Frequency from Nonuniform Samples,”
        SIAM Journal on Mathematical Analysis, Volume 43, pages 2078-2095, 10.1137/100798818, (2011)
        Definition 3.3 and 3.4
        '''

        w = Sfs.reshape(-1, 1) - torch.imag(dSx / Sx) / (2*self.pi)
        w = torch.abs(w)

        del dSx

        if gamma is None:
            gamma = np.sqrt(self.EPS64)
        elif gamma[0] == 'adaptive':
            gamma = torch_MAD(Sx) * 1.4826 * np.sqrt(2*np.log(self.signal_length)) * gamma[1]
        else:
            raise ValueError('gamma option {} not implemented; support `None` or `(`adaptive`, float)`')

        '''
        mark noise as inf to be later removed
        '''
        if self.visualizeFigs:
            self.visualize(w, dBscale=True, title='Phase-transformed result in dB scale')
        w[torch.abs(Sx) < gamma] = np.inf
        return w
    
    def replace_inf(self, x, inf_criterion):
        x = torch.where(torch.isinf(inf_criterion), torch.zeros_like(x), x)
        return x
    
    def synchrosqueeze(self, Sx, w, ssq_freqs, squeezetype='sum'):
        assert not (torch.min(w) < 0), 'max: {}; min: {}'.format(torch.max(w), torch.min(w))

        #########################################
        '''
        Amplitude manipulation on Sx
        '''
        if squeezetype == 'sum':
            Sx = Sx
        elif squeezetype == 'lebesque':
            Sx = torch.ones_like(Sx) / Sx.size(0)
        elif squeezetype == 'abs':
            Sx = torch.abs(Sx)
        else:
            raise ValueError('Unsupported squeeze function keyword; support `sum` or `lebesque` or `abs`.')
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

        if self.visualizeFigs:
            self.visualize(freq_mod_indices, dBscale=False, title='`Squeezed` frequency indices')

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

        Tx = indexed_sum((Sx * df).detach().cpu().numpy(), freq_mod_indices.detach().cpu().numpy())
        if self.time_run:
            torch.cuda.synchronize()
            print("Indexed sum time: %s seconds ---" % (time.time() - start_time))
        #########################################

        return Tx

    def visualize(self, T, flip=True, dBscale=True, title=None):
        if flip:
            T = np.flipud(T)
        if dBscale:
            T = 20*np.log10(np.abs(T)+1e-12)
        else:
            T = np.abs(T)

        if title is not None:
            plt.title(title)
        plt.imshow(T, aspect='auto', cmap='jet')
        plt.show()

    def sst_stft_forward(self, audio, sr, gamma=None, win_length=128, hop_length=1, use_Hann=False, visualize=True, time_run=True, squeezetype='sum'):
        assert hop_length == 1, 'Other hop length settings are not implemented, and lengths other than 1 do not comply with the invertibility criteria of SST.'
        n_fft = win_length
        self.visualizeFigs = visualize
        self.signal_length = int((audio.shape[0]//2)*2)
        self.time_run = time_run
        self.sr = sr
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft

        if self.time_run:
            sst_start_time = time.time()
        #########################################
        if self.time_run:
            torch.cuda.synchronize()
            start_time = time.time()
        '''
        Obtain STFTed signal and STFTed "difference" signal
        '''
        Sx, dSx = self.stft_dstft(audio, win_length, hop_length, n_fft, use_Hann)

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
        w = self.phase_transform(Sx, dSx, Sfs=Sfs, gamma=gamma)

        if self.time_run:
            torch.cuda.synchronize()
            print("Phase transform time: %s seconds ---" % (time.time() - start_time))
        #########################################

        #########################################
        '''
        Synchrosqueeze on the STFT
        '''
        # Tx is returned as numpy array
        Tx = self.synchrosqueeze(Sx, w, ssq_freqs=Sfs, squeezetype=squeezetype)
        Sx = Sx.detach().cpu().numpy()

        #########################################
        if self.time_run:
            print("--- SST total run time: %s seconds ---" % (time.time() - sst_start_time))

        if visualize:
            self.visualize(Tx, dBscale=True, title='SST result in dB scale')
            self.visualize(Sx, dBscale=True, title='STFT result in dB scale')
        return Tx, Sx
    
    def sst_stft_inverse(self, Tx):

        signal = Tx.real.sum(axis=0) * (2/self.window[len(self.window)//2].detach().cpu().numpy())

        return signal.real