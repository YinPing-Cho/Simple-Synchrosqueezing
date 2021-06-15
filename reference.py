import numpy as np
from ssqueezepy import ssq_cwt, issq_cwt, ssq_stft, issq_stft
from ssqueezepy._stft import get_window
import soundfile as sf
from utils import plot_signals, calc_SNR, pre_emphasis, zero_padding, make_frames, torch_MAD, find_closest, indexed_sum, de_zero_pad
import matplotlib.pyplot as plt
#np.set_printoptions(threshold=10_000)
def test_ssq_stft():
    """Same as `test_stft` except don't test `hop_len` or `modulated` since
    only `1` and `True` are invertible (by the library, and maybe theoretically).

    `window_scaling=.5` has >x2 greater MAE for some reason. May look into.
    """
    th = 1e-1
    for N in (128, 129):
      x = np.random.randn(N)
      for n_fft in (120, 121):
        for window_scaling in (1., .5):
          if window_scaling == 1:
              window = None
          else:
              window = get_window(window, win_len=n_fft//1, n_fft=n_fft)
              window *= window_scaling

          Sx, *_ = ssq_stft(x, window=window, n_fft=n_fft)
          xr = issq_stft(  Sx, window=window, n_fft=n_fft)

          txt = ("\nSSQ_STFT: (N, n_fft, window_scaling) = ({}, {}, {})"
                 ).format(N, n_fft, window_scaling)
          assert len(x) == len(xr), "%s != %s %s" % (N, len(xr), txt)
          mae = np.abs(x - xr).mean()
          assert mae < th, "MAE = %.2e > %.2e %s" % (mae, th, txt)

N = 2048
NyqF = N/2
time_len = 1
t = np.linspace(0, time_len, N*time_len, endpoint=False)
#xo = np.cos(2*np.pi*t*NyqF/4)
xo = np.sin(2*np.pi*np.sin(t)*np.cos(t/time_len)*NyqF/8)
xo += np.cos(2*np.pi*t*NyqF/2)
xo /= np.max(np.abs(xo))

noise = 0.3*np.random.standard_normal(N*time_len)  # add noise
x = xo + noise

print(x.shape)

original_signal = xo
input_signal = xo
sr = N

sf.write('input.wav', data=input_signal, samplerate=sr, subtype='PCM_16')

window = None
n_fft = 128

Tx, *_ = ssq_stft(xo, window=window, n_fft=n_fft)
T = np.abs(Tx)
#T = np.log(T+1e-3)
#T = T - np.min(T)
plt.imshow(T, aspect='auto', cmap='jet') # , vmin=0, vmax=.2
plt.show()
print('Tx max: {}; min: {}'.format(np.max(np.abs(Tx)), np.min(np.abs(Tx))))

recon_signal = issq_stft(  Tx, window=window, n_fft=n_fft)

#plt.plot(Tx)
#plt.show()



noise /= np.max(np.abs(recon_signal))
recon_signal /= np.max(np.abs(recon_signal))

#print('The SNR of recon signal: {} dB'.format(calc_SNR(recon_signal, noise)))

sf.write('rf_recon_Modi.wav', data=recon_signal, samplerate=sr, subtype='PCM_16')

signals = dict(Clean_Signal=xo, Input_Signal=input_signal, Reconstructed_Signal=recon_signal)
plot_signals(signals, samples=128, title="SST Before and After")