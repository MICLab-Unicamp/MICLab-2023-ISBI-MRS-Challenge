"""
Maintainer: Gabriel Dias (g172441@dac.unicamp.br)
            Mateus Oliveira (m203656@dac.unicamp.br)
"""

import numpy as np
from utils import normalized_stft


class PreProcessing:
    @staticmethod
    def spectrogram_channel(fid_off: np.ndarray, fid_on: np.ndarray, fs: np.float64,
                            larmorfreq: np.float64) -> np.ndarray:

        fid_diff = fid_on - fid_off
        fid_result = np.mean(fid_diff, axis=1)

        if fid_result.shape[0] == 2048:
            hop_size = 10
        elif fid_result.shape[0] == 4096:
            hop_size = 64

        spectrogram = normalized_stft(fid=fid_result,
                                      fs=fs,
                                      larmorfreq=larmorfreq,
                                      window_size=256,
                                      hop_size=hop_size)

        return spectrogram
