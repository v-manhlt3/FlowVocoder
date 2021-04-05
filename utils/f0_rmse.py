import librosa
import numpy as np
import os
import pyworld as pw
import matplotlib.pyplot as plt
import pysptk
from scipy.io import wavfile

ln10_inv = 1 / np.log(10)


def pad_to(x, target_len):
    pad_len = target_len - len(x)

    if pad_len <= 0:
        return x[:target_len]
    else:
        return np.pad(x, (0, pad_len), 'constant', constant_values=(0, 0))


def eval_snr(x_r, x_s):
    return 10 * np.log10(np.sum(x_s ** 2) / np.sum((x_s - x_r) ** 2))


def eval_MCD(x_r, x_s):
    c_r = librosa.feature.mfcc(x_r)
    c_s = librosa.feature.mfcc(x_s)

    temp = 2 * np.sum((c_r - c_s) ** 2, axis=0)
    # print(temp)
    return 10 * ln10_inv * (temp ** 0.5)


def plot_f0(*files, title=None):
    for file in files:

        if isinstance(file, tuple):
            file_path, label = file
        else:
            file_path = file
            label = None

        aud, sr = librosa.load(file_path, sr=None)
        f0 = pysptk.sptk.swipe(aud.astype(np.double), sr, hopsize=128)
        plt.plot(f0, label=label)

    plt.ylabel('f0(Hz)')
    plt.xlabel('frame')
    if title:
        plt.title(title)
    plt.legend(loc='upper right')
    plt.show()


def eval_rmse_f0(x_r, x_s, sr, frame_len='5', method='rapt', tone_shift=None):
    if method == 'harvest':
        f0_r, t = pw.harvest(x_r.astype(np.double), sr, frame_period=50)
        f0_s, t = pw.harvest(x_s.astype(np.double), sr, frame_period=50)
    elif method == 'dio':
        f0_r, t = pw.dio(x_r.astype(np.double), sr, frame_period=50)
        f0_s, t = pw.dio(x_s.astype(np.double), sr, frame_period=50)
    elif method == 'swipe':
        f0_r = pysptk.sptk.swipe(x_r.astype(np.double), sr, hopsize=128)
        f0_s = pysptk.sptk.swipe(x_s.astype(np.double), sr, hopsize=128)
    elif method == 'rapt':
        f0_r = pysptk.sptk.rapt(x_r.astype(np.float32), sr, hopsize=128)
        f0_s = pysptk.sptk.rapt(x_s.astype(np.float32), sr, hopsize=128)
    else:
        raise ValueError('no such f0 exract method')

    # length align
    f0_s = pad_to(f0_s, len(f0_r))

    # make unvoice / vooiced frame mask
    f0_r_uv = (f0_r == 0) * 1
    f0_r_v = 1 - f0_r_uv
    f0_s_uv = (f0_s == 0) * 1
    f0_s_v = 1 - f0_s_uv

    tp_mask = f0_r_v * f0_s_v
    tn_mask = f0_r_uv * f0_s_uv
    fp_mask = f0_r_uv * f0_s_v
    fn_mask = f0_r_v * f0_s_uv

    if tone_shift is not None:
        shift_scale = 2 ** (tone_shift / 12)
        f0_r = f0_r * shift_scale

    # only calculate f0 error for voiced frame
    y = 1200 * np.abs(np.log2(f0_r + f0_r_uv) - np.log2(f0_s + f0_s_uv))
    y = y * tp_mask
    # print(y.sum(), tp_mask.sum())
    f0_rmse_mean = y.sum() / tp_mask.sum()

    # only voiced/ unvoiced accuracy/precision
    vuv_precision = tp_mask.sum() / (tp_mask.sum() + fp_mask.sum())
    vuv_accuracy = (tp_mask.sum() + tn_mask.sum()) / len(y)

    return f0_rmse_mean, vuv_accuracy, vuv_precision


def eval_rmse_ap(x_r, x_s, sr, frame_len='5'):
    # TODO: find out what algorithm to use.  maybe pyworld d4c?
    pass


if __name__ == '__main__':

    import glob

    src_spk = "ground-truth"
    trg_spk = "flowvocoder"
    root_dir = "../Neural-Vocoder-Experiment/"

    src_files = glob.glob(os.path.join(root_dir, src_spk, "*.wav"))
    list_rmse_f0 = []

    for file_s in src_files:
        
        filename = os.path.basename(file_s)
        file_r = os.path.join(root_dir, trg_spk, filename)        

        aud_r, sr_r = librosa.load(file_r, sr=None)
        aud_s, sr_s = librosa.load(file_s, sr=None)

        sr, aud_r = wavfile.read(file_r)
        sr, aud_s = wavfile.read(file_s)

        assert sr_r == sr_s
        if len(aud_r) != len(aud_s):
            aud_r = aud_r[:len(aud_s)]
            aud_s = aud_s[:len(aud_r)]

        # mcd = eval_MCD(aud_r, aud_s)
        rmse_f0 = eval_rmse_f0(aud_r, aud_s, sr_r)
        list_rmse_f0.append(rmse_f0[0])
        print(rmse_f0)
    
    print("Root mean square error of fundamental frequency: ", np.array(list_rmse_f0).mean())

    
