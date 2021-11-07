import librosa
import numpy as np
import scipy
from scipy import signal
import argparse
import sys, os, time
import bss_eval
from subfunc import pesq2


target_fs = 16000
def wav_proc(proc_wav, ref_wav, nsy_wav):
    proc_sig, _ = librosa.load(proc_wav, mono = True, sr = target_fs)
    ref_sig,  _ = librosa.load(ref_wav,  mono = True, sr = target_fs)
    nsy_sig,  _ = librosa.load(nsy_wav,  mono = True, sr = target_fs)

    proc_sig_len = len(proc_sig)
    ref_sig_len  = len(ref_sig)
    nsy_sig_len  = len(nsy_sig)

    min_sig_len = np.min([ref_sig_len, proc_sig_len, nsy_sig_len])
    
    proc_sig = proc_sig[0:min_sig_len]; proc_sig = np.reshape(proc_sig, (1, len(proc_sig)))
    ref_sig  = ref_sig[0:min_sig_len];  ref_sig = np.reshape(ref_sig, (1, len(ref_sig)))
    nsy_sig  = nsy_sig[0:min_sig_len];  nsy_sig = np.reshape(nsy_sig, (1, len(nsy_sig)))
    
    (sdr, sir, sar, _) = bss_eval.bss_eval_sources(np.concatenate([ref_sig, nsy_sig - ref_sig]), np.concatenate([proc_sig, nsy_sig - proc_sig]), False)
    pesq = pesq2(ref_wav, proc_wav, sample_rate = 16000, program = '/home/data1/gwl/PESQ')
    
    sdr_4 = round(sdr[0], 4); sir_4 = round(sir[0], 4); sar_4 = round(sar[0], 4); pesq_4 = round(float(pesq), 4)
    result_print = 'SDR({}) SIR({}) SAR({}) PESQ({})'.format(sdr_4, sir_4, sar_4, pesq_4)
    print(result_print)
        
    return sdr[0], sir[0], sar[0], float(pesq)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--proc_dir', type = str, required = True, help = 'Proccessed wav directory.')
    parser.add_argument('--ref_dir', type = str, required = True, help = 'Clean wav directory.')
    parser.add_argument('--nsy_dir', type = str, required = True, help = 'Noisy wav directory.')
    parser.add_argument('--txt', type = str, required = True, help = 'result text')
    parser.add_argument('--snr', type = str, required = False, help = 'snr')
    
    args = parser.parse_args()
    
    sdr_total  = np.empty((1, )); sir_total  = np.empty((1, )); sar_total  = np.empty((1, )); pesq_total = np.empty((1, ))

    proc_dir = args.proc_dir
    ref_dir = args.ref_dir
    nsy_dir = args.nsy_dir
    measure_txt = args.txt
    
    if args.snr is not None:
        snr_cond = '_' + args.snr + '_'
    else:
        snr_cond = None
    
    cnt = 0
    f_txt = open(measure_txt,'w')
    
    for path, dirs, files in os.walk(proc_dir):
        for file in files:
            if (os.path.splitext(file)[1].lower() == '.wav') and (file.find(snr_cond) != -1):
#            if os.path.splitext(file)[1].lower() == '.wav':                    
                input_wav_path  = path + '/' + file
                sub_path = path.split(proc_dir)[-1]
                if sub_path is not None:
                    ref_wav_path  = ref_dir + '/' + file
                    nsy_wav_path  = nsy_dir + '/' + file
                else:
                    ref_wav_path  = ref_dir + '/' + sub_path + '/' + file
                    nsy_wav_path  = nsy_dir + '/' + sub_path + '/' + file
                
                print('Processing... ' + file, end = '\n')
                sdr_val, sir_val, sar_val, pesq_val = wav_proc(input_wav_path, ref_wav_path, nsy_wav_path)
                
                result_txt = '{}\t{}\t{}\t{}\n'.format(sdr_val, sir_val, sar_val, pesq_val)
                f_txt.write(result_txt)
                    
                sdr_total  = np.append(sdr_total,  sdr_val)
                sir_total  = np.append(sir_total,  sir_val)
                sar_total  = np.append(sar_total,  sar_val)
                pesq_total = np.append(pesq_total, pesq_val)


    #----final result
    sdr_mu = round(np.mean(sdr_total[1:]), 4)
    sdr_std = round(np.std(sdr_total[1:]), 4)
    sir_mu = round(np.mean(sir_total[1:]), 4)
    sir_std = round(np.std(sir_total[1:]), 4)
    sar_mu = round(np.mean(sar_total[1:]), 4)
    sar_std = round(np.std(sar_total[1:]), 4)
    pesq_mu = round(np.mean(pesq_total[1:]), 4)
    pesq_std = round(np.std(pesq_total[1:]), 4)

    result_print = 'SDR({}, {}) SIR({}, {}) SAR({}, {}) PESQ({}, {})'.format(
           sdr_mu, sdr_std, sir_mu, sir_std, sar_mu, sar_std, pesq_mu, pesq_std)
    print(result_print)

    result_txt = '{}, {}\n{}, {}\n{}, {}\n{}, {}'.format(
           sdr_mu, sdr_std, sir_mu, sir_std, sar_mu, sar_std, pesq_mu, pesq_std)
    
    f_txt.write(result_txt)
    f_txt.close()