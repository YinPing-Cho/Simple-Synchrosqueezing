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
        if len(audio.shape) != 1:
            audio = audio[:,0]
        dur = librosa.get_duration(y=audio, sr=sr)
        print('Audio name: {};\nSample rate: {};\nDuration: {};\nShape: {};'.format(filename, sr, dur, audio.shape))
        if pre_emph:
            audio = pre_emphasis(audio)

        original_signal = audio[:dur_limit*sr]

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
#audio_filename='500Hz_gong.flac'
#audio_filename='flute.flac'
#audio_filename='gong.flac'
#audio_filename='vibraphone.flac'
#audio_filename='piano.flac'
#audio_filename='soprano.flac'
#audio_filename='bass.flac'
#audio_filename='male.flac'
#audio_filename='violin.flac'
#audio_filename='1ksine.flac'

dur_limit = 10
pre_emph = False
noise_std = 0.2
write_Waves = False
original_signal, noised, sr = test_signal(use_synth=use_synth, audio_filename=audio_filename, dur_limit=dur_limit, pre_emph=pre_emph, noise_std=noise_std)

input_signal = noised

torch_device = 'cpu'
SST = Synchrosqueezing(torch_device=torch_device)

if use_synth and write_Waves:
    sf.write('synth_wave.wav', data=original_signal, samplerate=sr, subtype='PCM_16')
if write_Waves:
    sf.write('1k_d2.wav', data=input_signal, samplerate=sr, subtype='PCM_16')
Tx, Sx = SST.sst_stft_forward(audio=input_signal, sr=sr, gamma=('adaptive', 1.0), win_length=512, hop_length=1, use_Hann=True, visualize=True, time_run=True, squeezetype='sum')
print('Tx max: {}; min: {}'.format(np.max(np.abs(Tx)), np.min(np.abs(Tx))))

recon_signal = SST.sst_stft_inverse(Tx)

signals = dict(Clean_Signal=original_signal, Input_Signal=input_signal, Reconstructed_Signal=recon_signal)
plot_signals(signals, samples=int(0.25*sr), start=int(0.5*sr), title="SST Before and After")

recon_signal /= np.max(np.abs(recon_signal))

if write_Waves: 
    sf.write('recon_1k_d2.wav', data=recon_signal, samplerate=sr, subtype='PCM_16')