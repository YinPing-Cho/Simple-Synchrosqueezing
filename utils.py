import numpy as np
import torch

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
        
        if x.size(0) % 2 != 0:
            x = x[:-1]

        pad_each_side = (target_length - x.size(0)) / 2
        assert pad_each_side % 2 == 0
        pad_each_side = int(pad_each_side)

        x = torch.nn.functional.pad(x, (pad_each_side, pad_each_side))

        return x

def make_frames(x, frame_length, hop_length):
        num_frames = 1 + int(np.ceil((1.0 * x.size(0) - frame_length) / hop_length))
        indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames*hop_length, hop_length), (frame_length, 1)).T
        indices = np.array(indices, dtype=np.int32)

        # slice signal into frames
        frames = x[indices.T]
        return frames

def indexed_sum(a, k):
    """Sum `a` into rows of 2D array according to indices given by 2D `k`"""
    out = torch.zeros_like(a)
    for i in range(a.size(0)):
        for j in range(a.size(1)):
            out[k[i][j], j] += a[i][j]
    return out