"""
Maintainer: Gabriel Dias (g172441@dac.unicamp.br)
            Mateus Oliveira (m203656@dac.unicamp.br)
"""

import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from utils import pad_zeros_spectrogram, ReadDatasets
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
    model = SpectroViT().to(device)

    # Load the weights into the model
    load_dict = torch.load(args.weights)
    model.load_state_dict(load_dict["model_state_dict"])

    # Get the test data path
    test_data_path = args.test_data_path

    # Read spectrogram samples and corresponding tracks from the dataset
    input_transients, input_ppm, input_t = ReadDatasets.read_h5_sample_track_1(test_data_path)

    # Perform inference on each spectrogram sample
    pred_labels_stacked = None
    ppm_stacked = None
    for i in tqdm(range(input_transients.shape[0])):
        # Get the current spectrogram sample and corresponding tracks
        signal_input = input_transients[i, :, :, :]
        ppm = input_ppm[i, :]
        t = input_t[i, :]

        # Add a new dimension to the spectrogram sample
        signal_input = signal_input[np.newaxis, ...]

        # Generate the spectrogram
        spectrogram = PreProcessing.spectrogram(signal_input, t)

        # Pad zeros to the spectrogram
        spectrogram_padd = pad_zeros_spectrogram(spectrogram[0])
        spectrogram = spectrogram_padd[np.newaxis, ...]

        # Convert spectrogram and ppm to PyTorch tensors
        spectrogram = torch.from_numpy(spectrogram)
        ppm = torch.from_numpy(ppm)

        # Concatenate the spectrogram along the channel dimension
        three_channels_spectrogram = torch.cat([spectrogram, spectrogram, spectrogram])

        # Convert the input to the appropriate type and move it to the device
        inputs = three_channels_spectrogram.real.type(torch.FloatTensor)
        inputs = inputs.to(device)

        # Add an extra dimension to the input tensor
        inputs = torch.unsqueeze(inputs, dim=0)

        # Perform forward pass to get the predicted labels
        pred_labels = model(inputs)

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
