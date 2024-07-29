"""
Maintainer: Gabriel Dias (g172441@dac.unicamp.br)
            Mateus Oliveira (m203656@dac.unicamp.br)
"""

import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
from utils import ReadDatasets, zero_padding
from pre_processing import PreProcessing
from models import SpectroViT

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Predicted neural network MRS")
    parser.add_argument("weights", type=str, help="Weights neural network")
    parser.add_argument("test_data_path", type=str, help="Path to test dataset (.h5)")
    parser.add_argument("save_folder_path", type=str, help="add folder path which the predict .h5 file will be saved")
    args = parser.parse_args()

    # Determine device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = SpectroViT().to(device)

    # Load pre-trained weights
    load_dict = torch.load(args.weights)
    model.load_state_dict(load_dict["model_state_dict"])

    # Load test data
    test_data_path = args.test_data_path
    input_transients, input_ppm, input_t, larmorfreq = ReadDatasets.read_h5_sample_track_2(test_data_path)

    pred_labels_stacked = None
    ppm_stacked = None

    # Process each input sample
    for i in tqdm(range(input_transients.shape[0])):
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

        # Forward pass through the model
        pred_labels = pred_labels.detach().cpu().numpy()
        ppm = ppm.detach().cpu().numpy()

        # Normalize predicted labels
        pred_labels = (pred_labels - pred_labels.min()) / (pred_labels.max() - pred_labels.min())

        # Stack predicted labels and ppm values
        if i == 0:
            pred_labels_stacked = pred_labels
            ppm_stacked = ppm
        else:
            pred_labels_stacked = np.vstack((pred_labels_stacked, pred_labels))
            ppm_stacked = np.vstack((ppm_stacked, ppm))

    # Save the predicted labels and ppm values to a file
    save_path = os.path.join(args.save_folder_path, "track02.h5")
    ReadDatasets.write_h5_track2_predict_submission(
        filename=save_path,
        spectra_predict=pred_labels_stacked,
        ppm=ppm_stacked
    )
