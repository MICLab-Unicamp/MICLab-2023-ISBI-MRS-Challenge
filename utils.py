import yaml
import numpy as np
import h5py
from scipy import signal


def read_yaml(file: str) -> yaml.loader.FullLoader:
    """
    Read YAML configuration file and load the configurations.

    Args:
        file (str): Path to the YAML file.

    Returns:
        yaml.loader.FullLoader: Loaded configurations from the YAML file.
    """
    with open(file, "r") as yaml_file:
        configurations = yaml.load(yaml_file, Loader=yaml.FullLoader)

    return configurations


class ReadDatasets:
    @staticmethod
    def read_h5_sample_track_1(filename: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Read sample data from an H5 file for track 1.

        Args:
            filename (str): Name of the input H5 file.

        Returns:
            tuple: A tuple containing the transients, ppm, and t arrays.
        """
        with h5py.File(filename) as hf:
            ppm = hf["ppm"][()]
            t = hf["t"][()]
            transients = hf["transients"][()]

        return transients, ppm, t

    @staticmethod
    def read_h5_sample_track_2(filename: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Read sample data from an H5 file for track 2.

        Args:
            filename (str): Name of the input H5 file.

        Returns:
            tuple: A tuple containing the transients, ppm, and t arrays.
        """
        with h5py.File(filename) as hf:
            ppm = hf["ppm"][()]
            t = hf["t"][()]
            transients = hf['transient_fids'][()]

        return transients, ppm, t

    def read_h5_sample_track_3(filename: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray]:
        """
        Read sample data from an H5 file for track 3.

        Args:
            filename (str): Name of the input H5 file.

        Returns:
            tuple: A tuple containing the input transients, ppm, and t arrays for both downsampled and upsampled tracks.
        """
        with h5py.File(filename) as hf:
            input_ppm_down = hf['data_2048']["ppm"][()]
            input_t_down = hf['data_2048']["t"][()]
            input_transients_down = hf['data_2048']["transient_fids"][()]

            input_ppm_up = hf['data_4096']["ppm"][()]
            input_t_up = hf['data_4096']["t"][()]
            input_transients_up = hf['data_4096']["transient_fids"][()]

        return input_transients_down, input_ppm_down, input_t_down, \
            input_transients_up, input_ppm_up, input_t_up

    @staticmethod
    def write_h5_track1_predict_submission(filename: str,
                                           spectra_predict: np.ndarray,
                                           ppm: np.ndarray):
        """
        Write the predicted spectra and ppm values to an H5 file for track 1 submission.

        Args:
            filename (str): Name of the output H5 file.
            spectra_predict (ndarray): Predicted spectra.
            ppm (ndarray): PPM values associated with the predicted spectra.
        """
        with h5py.File(filename, "w") as hf:
            # Creating datasets in the H5 file
            hf.create_dataset("result_spectra", spectra_predict.shape, dtype=float, data=spectra_predict)
            hf.create_dataset("ppm", ppm.shape, dtype=float, data=ppm)

    @staticmethod
    def write_h5_track2_predict_submission(filename: str,
                                           spectra_predict: np.ndarray,
                                           ppm: np.ndarray):
        """
        Write the predicted spectra and ppm values to an H5 file for track 2 submission.

        Args:
            filename (str): Name of the output H5 file.
            spectra_predict (ndarray): Predicted spectra.
            ppm (ndarray): PPM values associated with the predicted spectra.
        """
        with h5py.File(filename, "w") as hf:
            # Creating datasets in the H5 file
            hf.create_dataset("result_spectra", spectra_predict.shape, dtype=float, data=spectra_predict)
            hf.create_dataset("ppm", ppm.shape, dtype=float, data=ppm)

    @staticmethod
    def write_h5_track3_predict_submission(filename: str,
                                           spectra_predict_down: np.ndarray,
                                           ppm_down: np.ndarray,
                                           spectra_predict_up: np.ndarray,
                                           ppm_up: np.ndarray):
        """
        Write the predicted spectra and ppm values to an H5 file for track 3 submission.

        Args:
            filename (str): Name of the output H5 file.
            spectra_predict_down (ndarray): Predicted spectra for the downsampled track3.
            ppm_down (ndarray): PPM values for the downsampled track3.
            spectra_predict_up (ndarray): Predicted spectra for the upsampled track3.
            ppm_up (ndarray): PPM values for the upsampled track3.
        """
        with h5py.File(filename, "w") as hf:
            # Creating datasets in the H5 file for downsampled track
            hf.create_dataset("result_spectra_2048", spectra_predict_down.shape, dtype=float,
                              data=spectra_predict_down)
            hf.create_dataset("ppm_2048", ppm_down.shape, dtype=float, data=ppm_down)

            # Creating datasets in the H5 file for upsampled track
            hf.create_dataset("result_spectra_4096", spectra_predict_up.shape, dtype=float,
                              data=spectra_predict_up)
            hf.create_dataset("ppm_4096", ppm_up.shape, dtype=float, data=ppm_up)


def generate_spectrogram(FID, t, window_size=256, hope_size=64, window='hann', nfft=None):
    """
    Generate a spectrogram from the given signal using Short-Time Fourier Transform (STFT).

    Args:
        FID (array-like): Input signal.
        t (array-like): Time array corresponding to the FID signal.
        window_size (int): Size of the analysis window (default: 256).
        hope_size (int): Size of the hop between windows (default: 64).
        window (str or tuple or array_like): Desired window to use (default: 'hann').
        nfft (int): Length of the FFT used, if a zero padded FFT is desired (default: None).

    Returns:
        Zxx (ndarray): 2D array containing the complex-valued spectrogram.
    """

    # calculating the overlap between the windows
    noverlap = window_size - hope_size

    # checking for the NOLA criterion
    if not signal.check_NOLA(window, window_size, noverlap):
        raise ValueError("Signal windowing fails Non-zero Overlap Add (NOLA) criterion; STFT not invertible")

    fs = 1 / (t[1] - t[0])

    # computing the STFT
    _, _, Zxx = signal.stft(np.real(FID), fs=fs, nperseg=window_size, noverlap=noverlap,
                            return_onesided=True, nfft=nfft)

    Zxx = Zxx / np.max(np.abs(Zxx))

    return Zxx


def pad_zeros_spectrogram(Zxx, output_shape=(224, 224)):
    # This function pads zeros to a spectrogram matrix to match a specified output shape.

    # Copy the input matrix to a new variable
    matrix = Zxx

    # Calculate the pad width for each dimension of the matrix
    pad_width = (
        (0, output_shape[0] - matrix.shape[0]),  # Pad rows with zeros
        (0, output_shape[1] - matrix.shape[1])  # Pad columns with zeros
    )

    # Pad the matrix with zeros using the calculated pad width
    padded_matrix = np.pad(matrix, pad_width, mode="constant")

    # Return the padded matrix
    return padded_matrix
