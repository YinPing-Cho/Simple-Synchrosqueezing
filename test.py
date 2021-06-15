from sst import Synchrosqueezing
import soundfile as sf
from ssqueezepy import ssq_cwt, issq_cwt, ssq_stft, issq_stft
import librosa
import torch
import numpy as np
from utils import pre_emphasis, plot_signals

def test_signal(use_synth, audio_filename='draw_16.wav', dur_limit=5, pre_emph=False, noise_std=1.0):
    if not use_synth:
        filename = audio_filename
        audio, sr = sf.read(filename, dtype='float32')
        dur = librosa.get_duration(y=audio, sr=sr)
        print('Audio name: {};\nSample rate: {};\nDuration: {};\nShape: {};'.format(filename, sr, dur, audio.shape))
        if pre_emph:
            audio = pre_emphasis(audio)

        original_signal = audio

    else:
        N = 16000
        NyqF = N/2
        time_len = dur_limit
        t = np.linspace(0, time_len, N*time_len, endpoint=False)
        xo = np.sin(2*np.pi*np.sin(t)*np.cos(t/time_len)*NyqF/2)
        xo /= np.max(np.abs(xo))
        xo += np.sin(2*np.pi*np.sin(t)*np.cos(t/time_len)*NyqF/4)
        xo += np.sin(2*np.pi*np.sin(t)*np.cos(t/time_len)*NyqF*3/4)
        xo /= np.max(np.abs(xo))

        print('Synth audio samples:', xo.shape)

        original_signal = xo
        sr = N

    original_signal /= np.max(np.abs(original_signal))
    noise = noise_std*np.random.standard_normal(len(original_signal))  # add noise

    return original_signal, original_signal + noise, sr

use_synth = False
audio_filename='draw_16.wav'
dur_limit = 5
pre_emph = False
noise_std = 0.0
original_signal, noised, sr = test_signal(use_synth=use_synth, audio_filename=audio_filename, dur_limit=dur_limit, pre_emph=pre_emph, noise_std=noise_std)

input_signal = noised

torch_device = 'cuda'
SST = Synchrosqueezing(torch_device=torch_device)

sf.write('original_clean.wav', data=original_signal, samplerate=sr, subtype='PCM_16')
sf.write('input_signal.wav', data=input_signal, samplerate=sr, subtype='PCM_16')
Tx, Sx = SST.sst_stft_forward(audio=input_signal, sr=sr, gamma=('adaptive', 1.0), win_length=512, hop_length=1, use_Hann=True, visualize=True, time_run=True)
print('Tx max: {}; min: {}'.format(np.max(np.abs(Tx)), np.min(np.abs(Tx))))

recon_signal = SST.sst_stft_inverse(Tx)


sf.write('recon_Modi.wav', data=recon_signal, samplerate=sr, subtype='PCM_16')

signals = dict(Clean_Signal=original_signal, Input_Signal=input_signal, Reconstructed_Signal=recon_signal)
plot_signals(signals, samples=sr//4, title="SST Before and After")