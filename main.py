import numpy as np
import soundfile as sf
import librosa
import librosa.display
import os, sys, argparse
import scipy
import scipy.io.wavfile
from scipy import signal
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
# import bss_eval

# MODIFY THE SAMPLE AND FILTER
sample_idx = 1 # 1 ~ 5
filter_mode = 'Lowpass' # lowpass, highpass, or bandpass

# DO NOT MODIFY PARAMETER!
fc = [500, 5000]
fs = 16000
len_filter = 512
if filter_mode.lower() == 'lowpass':
    coeff_b = signal.firwin(len_filter, [1, fc[0]], nyq = fs//2, pass_zero = False)
elif filter_mode.lower() == 'highpass':
    coeff_b = signal.firwin(len_filter, [fc[1], fs//2 - 1], nyq = fs//2, pass_zero = False)
elif filter_mode.lower() == 'bandpass':
    coeff_b = signal.firwin(len_filter, [fc[0], fc[1]], nyq = fs//2, pass_zero = False)
else:
    raise ValueError('Select \'lowpass\', \'highpass\', or \'bandpass\' instead of \'{}\''.format(filter_mode))
    
sig, _ = librosa.load(os.path.join(sample_mode, 'mixed_{}.wav'.format(sample_idx)), mono = True, sr = fs) # source audio file
rec_name = 'record_' + filter_mode.lower() + '.wav'
dis, _ = librosa.load(os.path.join(sample_mode, rec_name), mono = True, sr = fs) # record audio file
ref = signal.lfilter(coeff_b, 1, sig)

# DO NOT MODIFY BELOW CODE!
# Modify recorded audio file
# Remove silent
for th in np.array([80, 70, 60, 50, 40, 30]):
    dis_, on_off = librosa.effects.trim(dis, top_db = th)
    if (on_off[1] - on_off[0] > len(sig) * 0.65) and (on_off[1] - on_off[0] <= len(sig)):
        break
# Synchronize two signal
corr = fftconvolve(sig, dis_[::-1], mode = 'same')
delay = 2 * (len(corr)//2 - np.argmax(corr))
delay = np.max([delay, 0])
print('{:.2f} second delay in audio file'.format(delay / fs))
dis_rs = np.zeros(len(ref))
if len(dis_) + delay > len(ref):
    dis_rs[delay:delay + len(dis_[:len(ref) - delay])] = dis_[:len(ref) - delay]
    print('[warning] last {:.2f} second recored signal reducted due to recording environment'.format((len(dis_) - len(dis_[:len(ref) - delay])) / fs))
else:
    dis_rs[delay:delay + len(ref)] = dis_[:len(ref)]
    
fig = plt.figure(dpi = 50, figsize = (15, 18), facecolor = 'white')
fig.suptitle('{} analysis'.format(filter_mode), fontsize = 32)
ax = plt.subplot(321)
ax.set(title = 'Source signal waveform')
librosa.display.waveshow(sig, sr = fs, max_points = fs//2, x_axis = 'time')
plt.ylim([-0.7, 0.7])
ax = plt.subplot(322)
ax.set(title = 'Source signal spectrogram')
SIG = librosa.amplitude_to_db(np.abs(librosa.stft(sig)), ref = np.max)
librosa.display.specshow(SIG, y_axis = 'linear', x_axis = 'time', sr = fs)
ax = plt.subplot(323)
ax.set(title = 'Ideal signal waveform')
librosa.display.waveshow(ref, sr = fs, max_points = fs//2, x_axis = 'time')
plt.ylim([-0.7, 0.7])
ax = plt.subplot(324)
ax.set(title = 'Ideal signal spectrogram')
REF = librosa.amplitude_to_db(np.abs(librosa.stft(ref)), ref = np.max)
librosa.display.specshow(REF, y_axis = 'linear', x_axis = 'time', sr = fs)
ax = plt.subplot(325)
ax.set(title = 'Recorded signal waveform')
librosa.display.waveshow(dis_rs, sr = fs, max_points = fs//2, x_axis = 'time')
plt.ylim([-0.7, 0.7])
ax = plt.subplot(326)
ax.set(title = 'Recorded signal spectrogram')
DIS = librosa.amplitude_to_db(np.abs(librosa.stft(dis_rs)), ref = np.max)
librosa.display.specshow(DIS, y_axis = 'linear', x_axis = 'time', sr = fs)

# DO NOT MODIFY BELOW CODE
# Objective measure
def extract_overlapped_windows(x, nperseg, noverlap, window = None):
    # source: https://github.com/scipy/scipy/blob/v1.2.1/scipy/signal/spectral.py
    step = nperseg - noverlap
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//step, nperseg)
    strides = x.strides[:-1]+(step*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)
    if window is not None:
        result = window * result
    return result

def segSISNR(ref, dis ,fs = 16_000, frameLen = 0.1, overlap = 0.5):
    eps = np.finfo(np.float64).eps
    win_len = round(frame_len * fs)
    skip_len = int(np.floor((1. - overlap) * frame_len * fs))
    MIN_SNR, MAX_SNR= -10, 35 # minimum/maximum SNR in dB

    hannwin = 0.5 * (1. - np.cos(2 * np.pi * np.arange(1, win_len + 1) / (win_len + 1)))
    ref_frames = extract_overlapped_windows(ref, win_len, win_len - skip_len, hannwin)
    dis_frames = extract_overlapped_windows(dis, win_len, win_len - skip_len, hannwin)
    
    sig_e = np.power(ref_frames, 2).sum(-1)
    optimal_ratio = np.multiply(ref_frames, dis_frames).sum(-1) / (sig_e + eps)
    scaled_frames = optimal_ratio * ref_frames

    scaled_sig_e = np.power(scaled_frames, 2).sum(-1)
    noise_e = np.power(sig_e - dis_frames, 2).sum(-1)
    
    segsnr = 10. * np.log10(scaled_sig_e / (noise_e + eps) + eps)    
    segsnr[segsnr < MIN_SNR] = MIN_SNR
    segsnr[segmental_snr > MAX_SNR] = MAX_SNR
    segsnr = segsnr[:-1]
    return np.mean(segsnr)

# sdr, _, _, _ = bss_eval.bss_eval_sources(np.reshape(ref, (1, -1)), np.reshape(dis_rs, (1, -1)))
segsisnr = segSISNR(ref, dis)
print('Segmental scale-invariant signal-to-noise(segSISNR) score is {:.4f}'.format(segsisnr[0]))

# Save figure
fig.savefig('{}_{}_exp.png'.format(sample_mode, filter_mode), transparent = True)
ax.clear()
# Save reference signal
scipy.io.wavfile.write(os.path.join(sample_mode, '{}_reference.wav'.format(filter_mode)), fs, (np.iinfo(np.int16).max * ref.T).astype(np.int16))
