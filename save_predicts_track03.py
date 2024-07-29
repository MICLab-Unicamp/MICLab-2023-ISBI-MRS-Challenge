"""
Maintainer: Gabriel Dias (g172441@dac.unicamp.br)
            Mateus Oliveira (m203656@dac.unicamp.br)
"""

import argparse
import os
import torch
from utils import ReadDatasets, zero_padding
from pre_processing import PreProcessing
from tqdm import tqdm
import numpy as np
from models import SpectroViT, SpectroViTTrack3

if __name__ == "__main__":
    # Create an argument parser object
    parser = argparse.ArgumentParser(description="predicted neural network MRS")
    # Add command-line arguments
    parser.add_argument("weights_down", type=str, help="WEIGHTs neural network for the 2048 model")
    parser.add_argument("weights_up", type=str, help="WEIGHTs neural network for the 4096 model")
    parser.add_argument("test_data_path", type=str, help="add test path dataset .h5")
    parser.add_argument("save_folder_path", type=str, help="add folder path which the predict .h5 file will be saved")

    # Parse the command-line arguments
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Retrieve the test data path from the command-line arguments
    test_data_path = args.test_data_path

    # Read the necessary datasets from the .h5 file
    input_transients_down, input_ppm_down, input_t_down, input_transients_up, input_ppm_up, input_t_up, larmorfreq = \
        ReadDatasets.read_h5_sample_track_3(test_data_path)

    # Create dictionaries to store configurations, models, and load dictionaries
    models = {}
    load_dicts = {}

    # Configurations and models for the 2048 model
    models["down"] = SpectroViT()
    load_dicts["down"] = torch.load(args.weights_down)
    models["down"].load_state_dict(load_dicts["down"]["model_state_dict"])
    models["down"].to(device)
    models["down"].eval()

    # Configurations and models for the 4096 model
    models["up"] = SpectroViTTrack3()
    load_dicts["up"] = torch.load(args.weights_up)
    models["up"].load_state_dict(load_dicts["up"]["model_state_dict"])
    models["up"].to(device)
    models["up"].eval()

    # Create dictionaries to store the stacked predicted labels and ppm values
    pred_labels_stacked = {}
    ppm_stacked = {}

    # Loop over the models
    for model_type in ["down", "up"]:
        # Select the appropriate input data based on the model type
        input_transients = input_transients_down if model_type == "down" else input_transients_up
        input_ppm = input_ppm_down if model_type == "down" else input_ppm_up
        input_t = input_t_down if model_type == "down" else input_t_up

        # Initialize the stacked predicted labels and ppm values
        pred_labels_stacked[model_type] = None
        ppm_stacked[model_type] = None

        print(f"Model Inference for the {model_type} sampled dataset")
        # Loop over the input transients
        for i in tqdm(range(input_transients.shape[0])):
            # Retrieve the current transient, ppm, and t values
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

            model = models[model_type].to(device)
            # Get the predicted labels from the model
            pred_labels = model(three_channels_spectrogram)

            # Convert the predicted labels and ppm to NumPy arrays
            pred_labels = pred_labels.detach().cpu().numpy()
            ppm = ppm.detach().cpu().numpy()

            # Perform normalization on the predicted labels
            pred_labels = (pred_labels - pred_labels.min()) / (pred_labels.max() - pred_labels.min())

            # Stack the predicted labels and ppm values
            if i == 0:
                pred_labels_stacked[model_type] = pred_labels
                ppm_stacked[model_type] = ppm
            else:
                pred_labels_stacked[model_type] = np.vstack((pred_labels_stacked[model_type], pred_labels))
                ppm_stacked[model_type] = np.vstack((ppm_stacked[model_type], ppm))

    # Define the save file path
    save_path= os.path.join(args.save_folder_path, "track03.h5")

    # Write the predicted labels and ppm values to an .h5 file using the ReadDatasets class method
    ReadDatasets.write_h5_track3_predict_submission(filename=save_path,
                                                    spectra_predict_down=pred_labels_stacked["down"],
                                                    ppm_down=ppm_stacked["down"],
                                                    spectra_predict_up=pred_labels_stacked["up"],
                                                    ppm_up=ppm_stacked["up"])
