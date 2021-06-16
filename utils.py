import numpy as np
import torch
from numba import jit, prange
import matplotlib.pyplot as plt

def plot_signals(signals, samples=None, start=0, title=None):
    for name, s in signals.items():
        if samples is None:
            x_labels = np.linspace(0, s.shape[0], s.shape[0])
            plt.plot(x_labels, s, label=name)
        else:
            assert samples + start <= s.shape[0]
            x_labels = np.linspace(start, start+samples, samples)
            plt.plot(x_labels, s[start:start+samples], label=name)
    
    if title is not None:
        plt.title(title)
    plt.legend()
    plt.show()
    return None

def signaltonoise_dB(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return 20*np.log10(abs(np.where(sd == 0, 0, m/sd)))

@jit(nopython=True, cache=True, parallel=True)
def pre_emphasis(signal, coefficient = 0.95):
    return np.append(signal[0],signal[1:] - coefficient*signal[:-1])

@jit(nopython=True, cache=True)
def is_power_of_2(num):
    return (num & (num-1) == 0) and num != 0

def smallest_greater_pow2(num):
    return 1<<(num-1).bit_length()

def make_frames(x, frame_length, hop_length):
    num_frames = 1 + int(np.ceil((x.size(0) - frame_length) / hop_length))
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames*hop_length, hop_length), (frame_length, 1)).T
    indices = np.array(indices, dtype=np.int32)

    # slice signal into frames
    frames = x[indices.T]
    return frames

@torch.jit.script
def torch_MAD(x):
    x = x.abs()
    return torch.median(torch.abs(x - torch.median(x)))

@torch.jit.script
def find_closest(a, v):
    """Equivalent to argmin(abs(a[i, j] - v)) for all i, j; a is 2D, v is 1D.
    Credit: Divakar -- https://stackoverflow.com/a/64526158/10133797
    """
    sidx = v.argsort()
    v_s = v[sidx]
    idx = torch.searchsorted(v_s, a)
    idx[idx == len(v)] = len(v) - 1
    idx0 = (idx-1).clip(min=0)

    m = torch.abs(a - v_s[idx]) >= torch.abs(v_s[idx0] - a)
    m[idx == 0] = 0
    idx[m] -= 1
    out = sidx[idx]
    return out

def indexed_sum(a, k):
    """Sum `a` into rows of 2D array according to indices given by 2D `k`."""
    out = np.zeros(a.shape, dtype=a.dtype)
    _parallel_indexed_sum(a, k, out)
    return out
@jit(nopython=True, cache=True, parallel=True)
def _parallel_indexed_sum(a, k, out):
    for j in prange(a.shape[1]):
        for i in prange(a.shape[0]):
            out[k[i, j], j] += a[i, j]