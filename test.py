from sst import Synchrosqueezing
import soundfile as sf
from ssqueezepy import ssq_cwt, issq_cwt, ssq_stft, issq_stft
import librosa
import torch
import numpy as np
from utils import pre_emphasis, plot_signals
import csv
import os
import argparse

def test_signal(use_synth, audio_filename=None, dur_limit=5, pre_emph=False, noise_std=1.0, fix_sr=None):
    if not use_synth:
        filename = audio_filename
        audio, sr = sf.read(filename, dtype='float32')
        if len(audio.shape) != 1:
            audio = audio[:,0]
        dur = librosa.get_duration(y=audio, sr=sr)
        #print('Audio name: {};\nSample rate: {};\nDuration: {};\nShape: {};'.format(filename, sr, dur, audio.shape))
        if pre_emph:
            audio = pre_emphasis(audio)

        original_signal = audio[:int(dur_limit*sr)]

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

        #print('Synth audio samples:', xo.shape)

        original_signal = xo
        sr = N

    if fix_sr is not None:
        original_signal = librosa.resample(original_signal, sr, fix_sr)

    original_signal /= np.max(np.abs(original_signal))
    noise = noise_std*np.random.standard_normal(len(original_signal))  # add noise

    return original_signal, original_signal + noise, sr

def calc_SNR(original_signal, corrupted_signal):
    signal = original_signal
    noise = corrupted_signal - original_signal

    signal_energy = np.mean(np.power(signal, 2))
    noise_energy = np.mean(np.power(noise, 2))

    return 10 * np.log10(signal_energy / noise_energy)

def calc_RMSE(original_signal, corrupted_signal):
    diff = corrupted_signal - original_signal
    RMSE = np.sqrt(np.mean(np.power(diff, 2)))

    return RMSE

def sst_cycle(torch_device, sr, filename, original_signal, noised_signal, visualize, outdir='TEST', write_Waves=False):

    print('\nTesting... Audio name: {};\nSample rate: {};Shape: {};'.format(filename, sr, original_signal.shape))
    filename = filename.split('.')[0]
    SST = Synchrosqueezing(torch_device=torch_device)

    noised_SNR = calc_SNR(original_signal, noised_signal)
    noised_RMSE = calc_RMSE(original_signal, noised_signal)

    if write_Waves:
        sf.write(os.path.join(outdir,filename+'_original.wav'), data=original_signal, samplerate=sr, subtype='PCM_16')
        sf.write(os.path.join(outdir,filename+'_noised.wav'), data=noised_signal, samplerate=sr, subtype='PCM_16')

    ### Test with `clean` signal
    Tx, Sx = SST.sst_stft_forward(audio=original_signal, sr=sr, gamma=('adaptive', 1.0), win_length=512, hop_length=1, use_Hann=True, visualize=visualize, time_run=True, squeezetype='sum')
    print('Tx max: {}; min: {}'.format(np.max(np.abs(Tx)), np.min(np.abs(Tx))))
    recon_signal = SST.sst_stft_inverse(Tx)
    clean_recon_SNR = calc_SNR(original_signal, recon_signal)
    clean_recon_RMSE = calc_RMSE(original_signal, recon_signal)

    if visualize:
        signals = dict(Clean_Signal=original_signal, Reconstructed_Signal=recon_signal)
        plot_signals(signals, samples=int(0.25*sr), start=int(0.5*sr), title="SST Before and After")

    if write_Waves:
        recon_signal /= np.max(np.abs(recon_signal))
        sf.write(os.path.join(outdir,filename+'_recon_original.wav'), data=recon_signal, samplerate=sr, subtype='PCM_16')
    
    ### Test with `noised` signal
    Tx, Sx = SST.sst_stft_forward(audio=noised_signal, sr=sr, gamma=('adaptive', 1.0), win_length=512, hop_length=1, use_Hann=True, visualize=visualize, time_run=True, squeezetype='sum')
    print('Tx max: {}; min: {}'.format(np.max(np.abs(Tx)), np.min(np.abs(Tx))))
    recon_signal = SST.sst_stft_inverse(Tx)
    noised_recon_SNR = calc_SNR(original_signal, recon_signal)
    noised_recon_RMSE = calc_RMSE(original_signal, recon_signal)

    if visualize:
        signals = dict(Clean_Signal=original_signal, Reconstructed_Signal=recon_signal)
        plot_signals(signals, samples=int(0.25*sr), start=int(0.5*sr), title="SST Before and After")
        signals = dict(Noised_Signal=noised_signal, Reconstructed_Signal=recon_signal)
        plot_signals(signals, samples=int(0.25*sr), start=int(0.5*sr), title="Clean vs Noised signals")

    if write_Waves:
        recon_signal /= np.max(np.abs(recon_signal))
        sf.write(os.path.join(outdir,filename+'_recon_noised.wav'), data=recon_signal, samplerate=sr, subtype='PCM_16')

    print('Noised SNR: {}; Clean Recon SNR: {}; Noised Recon SNR: {}'.format(noised_SNR, clean_recon_SNR, noised_recon_SNR))
    print('Noised RMSE: {}; Clean Recon RMSE: {}; Noised Recon RMSE: {}'.format(noised_RMSE, clean_recon_RMSE, noised_recon_RMSE))
    return noised_SNR, clean_recon_SNR, noised_recon_SNR, noised_RMSE, clean_recon_RMSE, noised_recon_RMSE

def main(args):
    if not os.path.isdir(args.outdir):
        os.mkdir(args.outdir)
    out_csv_path = os.path.join(args.outdir, 'SNR_test_results.csv')
    with open(out_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['TestSampleName','NoisedInputSignalSNR','ReconFromCleanSNR','ReconFromNoisedSNR','NoisedSignalRMSE','ReconFromCleanRMSE','ReconFromNoisedRMSE'])

        for filename in args.filenamelist:
            if filename == 'synth':
                use_synth = True
            else:
                use_synth = False

            dur_limit = args.audio_parameters['dur_limit']
            pre_emph = args.audio_parameters['pre_emph']
            noise_std = args.audio_parameters['noise_std']
            fix_sr = args.audio_parameters['fix_sr']
            original_signal, noised_signal, sr = \
                test_signal(use_synth=use_synth, audio_filename=filename, dur_limit=dur_limit, pre_emph=pre_emph, noise_std=noise_std, fix_sr=fix_sr)
            
            noised_SNR, clean_recon_SNR, noised_recon_SNR, noised_RMSE, clean_recon_RMSE, noised_recon_RMSE = \
                sst_cycle(args.torch_device, sr, filename, original_signal, noised_signal, args.visualize, outdir=args.outdir, write_Waves=args.write_waves)
            writer.writerow([filename,noised_SNR,clean_recon_SNR,noised_recon_SNR,noised_RMSE,clean_recon_RMSE,noised_recon_RMSE])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-td', '--torch_device', type=str, default='cpu',
                        help='`cpu` or `cuda`')
    parser.add_argument('-v', '--visualize', type=bool, default=True,
                        help='generate plots or not')
    parser.add_argument('-w', '--write_waves', type=bool, default=True,
                        help='dump wave files or not')
    parser.add_argument('-od', '--outdir', type=str, default='TEST',
                        help='directory where wave files are to be stored')
    parser.add_argument('-fnmlist', '--filenamelist', type=list, default=[
        'synth', 'male.flac','flute.flac'
        ], help='list of test file names')
    '''
    'synth', 'draw_16.wav','500Hz_gong.flac','flute.flac','gong.flac','vibraphone.flac','piano.flac','soprano.flac',\
        'bass.flac','male.flac','violin.flac','1ksine.flac'
    '''
    parser.add_argument('-ap', '--audio_parameters', type=dict, default={
        'dur_limit':5.0,
        'pre_emph':False,
        'noise_std':0.2,
        'fix_sr':None,
    }, help='test audio parameters')

    args = parser.parse_args()

    main(args)
