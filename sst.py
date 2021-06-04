import librosa
import torch
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from utils import zero_padding, make_frames, indexed_sum

class Synchrosqueezing:
    def __init__(self, torch_device):
        self.torch_device = torch_device
        torch.set_grad_enabled(False)

        self.pi = np.pi
        self.EPS32 = np.finfo(np.float32).eps
        self.EPS64 = np.finfo(np.float64).eps

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
        window = torch.hann_window(window_length=win_length, device=torch_device)
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
        ffted_dSx = torch.fft.rfft(dSx, dim=0)

        return ffted_Sx, ffted_dSx

    def get_Sfs(self, Sx, sr):
        Sfs = torch.linspace(0, .5*sr, Sx.size(0), device=Sx.device)
        return Sfs

    def phase_transform(self, Sx, dSx, Sfs, gamma):
        w = Sfs.reshape(-1, 1) - torch.imag(dSx / Sx) / (2*self.pi)
        w[torch.abs(Sx) < gamma] = np.inf
        return w

    def replace_inf(self, x, inf_criterion, constant=0):
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if np.isinf(inf_criterion[i][j]):
                    x[i][j] = constant
        return x

    def find_closest(self, a, v):
        """Equivalent to argmin(abs(a[i, j] - v)) for all i, j; a is 2D, v is 1D.
        Credit: Divakar -- https://stackoverflow.com/a/64526158/10133797
        """
        sidx = v.argsort()
        v_s = v[sidx]
        idx = np.searchsorted(v_s, a)
        idx[idx == len(v)] = len(v) - 1
        idx0 = (idx-1).clip(min=0)

        m = np.abs(a - v_s[idx]) >= np.abs(v_s[idx0] - a)
        m[idx == 0] = 0
        idx[m] -= 1
        out = sidx[idx]
        return out

    def synchrosqueeze(self, Wx, w, ssq_freqs, squeezetype='lebesque'):
        assert not (torch.min(w) < 0), 'max: {}; min: {}'.format(torch.max(w), torch.min(w))

        if squeezetype == 'lebesque':
            Wx = torch.ones_like(Wx) / Wx.size(0)
        elif squeezetype == 'abs':
            Wx = torch.abs(Wx)
        else:
            raise ValueError('Unsupported squeeze function keyword; support `lebesque` or `abs`.')
        
        Wx = self.replace_inf(Wx, inf_criterion=w)

        # squeeze by spectral adjacency
        freq_mod_indices = self.find_closest(w, ssq_freqs)

        df = ssq_freqs[1] - ssq_freqs[0]
        Tx = indexed_sum(Wx * df, freq_mod_indices)

        return Tx
    
    def audio_preparation(self, audio, sr):
        audio = torch.tensor(audio, device=torch_device)
        time_indices = torch.linspace(start=0,end=dur,steps=audio.size(0), device=torch_device)

        dt = time_indices[1]-time_indices[0]
        assert torch.isclose(dt,dur/torch.tensor(audio.size(0)),atol=1e-5), '{} and {}'.format(dt, dur/audio.size(0))
        audio = zero_padding(audio)

        return audio, dt

    def visualize(self, T, ):
        plt.imshow(torch.abs(T), aspect='auto', cmap='jet')
        plt.show()

    def sst_stft_forward(self, audio, sr, gamma=None, win_length=512, hop_length=256, n_fft=1024, visualize=True):
        audio, dt = self.audio_preparation(audio, sr)

        Sx, dSx = self.stft_dstft(audio, win_length, hop_length, n_fft)
        Sfs = self.get_Sfs(Sx, sr)

        if gamma is None:
            gamma = np.sqrt(np.sqrt(self.EPS32))
        w = self.phase_transform(Sx, dSx, Sfs, gamma)

        Tx = self.synchrosqueeze(Sx, w, ssq_freqs=Sfs, squeezetype='lebesque')

        if visualize:
            self.visualize(Tx)
            self.visualize(Sx)
        return Tx, Sx

torch_device = 'cpu'
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
Tx, Sx = SST.sst_stft_forward(audio=audio, sr=N, gamma=3, visualize=True)

plt.plot(Tx)
plt.show()