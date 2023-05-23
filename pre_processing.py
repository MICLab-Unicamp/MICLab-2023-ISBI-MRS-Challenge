import numpy as np
from utils import generate_spectrogram


class PreProcessingPipelineBaseline:
    @staticmethod
    def subtract_spectrum_s(signal_1: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Computation to generate the spectrogram.

        Args:
            signal_1 (np.ndarray): Input signal array.
            t (np.ndarray): Time array.

        Returns:
            np.ndarray: Result spectrogram.
        """
        if len(signal_1.shape) == 3:
            signal_1 = np.fft.fftshift(np.fft.ifft(signal_1, axis=1), axes=1)
            return signal_1[:, :, 1] - signal_1[:, :, 0]

        if len(signal_1.shape) == 4:
            fid_on = signal_1.mean(axis=3)[:, :, 1]
            fid_off = signal_1.mean(axis=3)[:, :, 0]

            fid_result = fid_on - fid_off

            if signal_1.shape[1] == 2048:
                signal_spectrum = generate_spectrogram(fid_result, t,
                                                       window_size=256,
                                                       hope_size=10,
                                                       nfft=446)
            else:
                signal_spectrum = generate_spectrogram(fid_result, t)

            return signal_spectrum
