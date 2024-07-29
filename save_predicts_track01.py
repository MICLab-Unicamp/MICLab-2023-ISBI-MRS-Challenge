"""
Maintainer: Gabriel Dias (g172441@dac.unicamp.br)
            Mateus Oliveira (m203656@dac.unicamp.br)
"""

import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from utils import zero_padding, ReadDatasets
from pre_processing import PreProcessing
from models import SpectroViT

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="predicted neural network MRS")
    parser.add_argument("weights", type=str, help="WEIGHTs neural network")
    parser.add_argument("test_data_path", type=str, help="add test path dataset .h5")
    parser.add_argument("save_folder_path", type=str, help="add folder path which the predict .h5 file will be saved")
    args = parser.parse_args()

    # Check if CUDA is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = SpectroViT()

    # Load the weights into the model
    load_dict = torch.load(args.weights)
    model.load_state_dict(load_dict["model_state_dict"])
    model.to(device)
    model.eval()

    # Get the test data path
    test_data_path = args.test_data_path

    # Read spectrogram samples and corresponding tracks from the dataset
    input_transients, input_ppm, input_t, larmorfreq = ReadDatasets.read_h5_sample_track_1(test_data_path)

    # Perform inference on each spectrogram sample
    pred_labels_stacked = None
    ppm_stacked = None
    for i in tqdm(range(input_transients.shape[0])):
        # Get the current spectrogram sample and corresponding tracks
        signal_input = input_transients[i, :, :, :]
        ppm = input_ppm[i, :]
        t = input_t[i, :]

        fs = np.float64(1 / (t[1] - t[0]))
        fid_off, fid_on = signal_input[:, 0, :], signal_input[:, 1, :]

        spectrogram1 = PreProcessing.spectrogram_channel(fid_off=fid_off[:, 0:14],
                                                         fid_on=fid_on[:, 0:14],
                                                         fs=fs,
                                                         larmorfreq=larmorfreq)
        spectrogram2 = PreProcessing.spectrogram_channel(fid_off=fid_off[:, 14:27],
                                                         fid_on=fid_on[:, 14:27],
                                                         fs=fs,
                                                         larmorfreq=larmorfreq)
        spectrogram3 = PreProcessing.spectrogram_channel(fid_off=fid_off[:, 27:40],
                                                         fid_on=fid_on[:, 27:40],
                                                         fs=fs,
                                                         larmorfreq=larmorfreq)

        spectrogram1 = zero_padding(spectrogram1)
        spectrogram1 = spectrogram1[np.newaxis, ...]
        spectrogram1 = torch.from_numpy(spectrogram1.real)

        spectrogram2 = zero_padding(spectrogram2)
        spectrogram2 = spectrogram2[np.newaxis, ...]
        spectrogram2 = torch.from_numpy(spectrogram2.real)

        spectrogram3 = zero_padding(spectrogram3)
        spectrogram3 = spectrogram3[np.newaxis, ...]
        spectrogram3 = torch.from_numpy(spectrogram3.real)

        ppm = torch.from_numpy(ppm)

        three_channels_spectrogram = torch.concat([spectrogram1, spectrogram2, spectrogram3])
        three_channels_spectrogram = three_channels_spectrogram[np.newaxis, ...]

        three_channels_spectrogram = three_channels_spectrogram.type(torch.FloatTensor).to(device)
        # Perform forward pass to get the predicted labels
        pred_labels = model(three_channels_spectrogram)

        # Convert the predicted labels and ppm to NumPy arrays
        pred_labels = pred_labels.detach().cpu().numpy()
        ppm = ppm.detach().cpu().numpy()

        # Perform normalization on the predicted labels
        pred_labels = (pred_labels - pred_labels.min()) / (pred_labels.max() - pred_labels.min())

        # Stack the predicted labels and ppm for all samples
        if i == 0:
            pred_labels_stacked = pred_labels
            ppm_stacked = ppm
        else:
            pred_labels_stacked = np.vstack((pred_labels_stacked, pred_labels))
            ppm_stacked = np.vstack((ppm_stacked, ppm))

    # Define the path to save the predicted results
    save_path = os.path.join(args.save_folder_path, "track01.h5")

    # Write the predicted spectra and ppm to the output file
    ReadDatasets.write_h5_track1_predict_submission(filename=save_path,
                                                    spectra_predict=pred_labels_stacked,
                                                    ppm=ppm_stacked)
