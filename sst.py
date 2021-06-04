import librosa
import torch
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

torch_device = 'cpu'
torch.set_grad_enabled(False)

pi = np.pi
EPS32 = np.finfo(np.float32).eps
EPS64 = np.finfo(np.float64).eps

def is_power_of_2(num):
    return (num & (num-1) == 0) and num != 0

def smallest_greater_pow2(num):
    return 1<<(num-1).bit_length()

def zero_padding(x, target_length=None):
    if target_length is not None:
        assert target_length > x.size(0), '{} and {}'.foramt(target_length, x.size(0))
        assert is_power_of_2(target_length)
    else:
        target_length = smallest_greater_pow2(x.size(0))
    print(target_length)
    
    if x.size(0) % 2 != 0:
        x = x[:-1]
    
    print(x.size(0))

    pad_each_side = (target_length - x.size(0)) / 2
    assert pad_each_side % 2 == 0
    pad_each_side = int(pad_each_side)

    x = torch.nn.functional.pad(x, (pad_each_side, pad_each_side))
    print(x.size(0))

    return x

def xi_function(scale, N):
    # from: https://github.com/OverLordGoldDragon/ssqueezepy/blob/efc6f916974be6459f92b35fb21d17b3d7553ed5/ssqueezepy/wavelets.py#L473
    """N=128: [0, 1, 2, ..., 64, -63, -62, ..., -1] * (2*pi / N) * scale
       N=129: [0, 1, 2, ..., 64, -64, -63, ..., -1] * (2*pi / N) * scale
    """
    xi = torch.zeros(N, device=torch_device)
    h = scale * (2 * pi) / N
    for i in range(N // 2 + 1):
        xi[i] = i * h
    for i in range(N // 2 + 1, N):
        xi[i] = (i - N) * h
    return xi

def win_and_diff_win(win_length, n_fft):
    window = torch.hann_window(window_length=win_length, device=torch_device)
    window = zero_padding(window, n_fft)
    
    ffted_win = torch.fft.fft(window)
    n_win = len(window)
    xi = xi_function(1, n_win)
    if n_win % 2 == 0:
        xi[n_win // 2] = 0
    diffed_win = torch.fft.ifft(ffted_win * 1j * xi).real

    return window, diffed_win

def make_frames(x, frame_length, hop_length):
    num_frames = 1 + int(np.ceil((1.0 * x.size(0) - frame_length) / hop_length))
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames*hop_length, hop_length), (frame_length, 1)).T
    indices = np.array(indices,dtype=np.int32)

    # slice signal into frames
    frames = x[indices]
    return frames.T

def stft_dstft(audio, win_length, hop_length, n_fft):
    window, diffed_window = win_and_diff_win(win_length, n_fft)
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

    print(Sx.size())
    print(dSx.size())

    # n_fft*num_frames -> (b_fft//2+1)*num_frames
    ffted_Sx = torch.fft.rfft(Sx, dim=0)
    ffted_dSx = torch.fft.rfft(dSx, dim=0)

    print(ffted_Sx.size())
    print(ffted_dSx.size())

    gamma = np.sqrt(EPS32)

    return ffted_Sx, ffted_dSx

def get_Sfs(Sx, sr):
    Sfs = torch.linspace(0, .5*sr, Sx.size(0), device=Sx.device)
    return Sfs

def phase_transform(Sx, dSx, SfS, gamma):
    w = Sfs.reshape(-1, 1) - torch.imag(dSx / Sx) / (2*pi)
    w[np.abs(Sx) < gamma] = np.inf
    return w

def sst_stft_forward(audio, sr, dt, win_length=512, hop_length=256, n_fft=1024):
    Sx, dSx = stft_dstft(audio, win_length, hop_length, n_fft)
    Sfs = get_Sfs(Sx, sr)
    print(Sfs.size())
    return

filename = 'draw_16.wav'
audio, sr = sf.read(filename, dtype='float32')
dur = librosa.get_duration(y=audio, sr=sr)
print('Audio name: {};\nSample rate: {};\nDuration: {};\nShape: {};'.format(filename, sr, dur, audio.shape))
#plt.plot(audio)
#plt.show()

audio = torch.tensor(audio, device=torch_device)
time_indices = torch.linspace(start=0,end=dur,steps=audio.size(0), device=torch_device)
print(time_indices)

dt = time_indices[1]-time_indices[0]
assert torch.isclose(dt,dur/torch.tensor(audio.size(0))), '{} and {}'.format(dt, dur/audio.size(0))

audio = zero_padding(audio)

sst_stft_forward(audio, sr, dt)