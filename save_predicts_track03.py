import argparse
import os
import torch
from utils import ReadDatasets, pad_zeros_spectrogram, read_yaml
from pre_processing import PreProcessingPipelineBaseline
from tqdm import tqdm
import numpy as np
from models import TimmSimpleCNN, TimmSimpleCNNTrack3

if __name__ == "__main__":
    # Create an argument parser object
    parser = argparse.ArgumentParser(description="predicted neural network MRS")
    # Add command-line arguments
    parser.add_argument("config_file_down", type=str, help="config neural network yaml for the 2048 model")
    parser.add_argument("config_file_up", type=str, help="config neural network yaml for the 4096 model")
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
    input_transients_down, input_ppm_down, input_t_down, input_transients_up, input_ppm_up, input_t_up = \
        ReadDatasets.read_h5_sample_track_3(test_data_path)

    # Create dictionaries to store configurations, models, and load dictionaries
    configs = {}
    models = {}
    load_dicts = {}

    # Configurations and models for the 2048 model
    configs["down"] = read_yaml(args.config_file_down)
    models["down"] = TimmSimpleCNN(**configs["down"]["model"]["TimmSimpleCNN"])
    load_dicts["down"] = torch.load(args.weights_down)
    models["down"].load_state_dict(load_dicts["down"]["model_state_dict"])

    # Configurations and models for the 4096 model
    configs["up"] = read_yaml(args.config_file_up)
    models["up"] = TimmSimpleCNNTrack3(**configs["up"]["model"]["TimmSimpleCNNTrack3"])
    load_dicts["up"] = torch.load(args.weights_up)
    models["up"].load_state_dict(load_dicts["up"]["model_state_dict"])

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
            fid_noise = input_transients[i, :, :, :]
            ppm = input_ppm[i, :]
            t = input_t[i, :]

            # Add a new axis to the fid_noise array
            fid_noise = fid_noise[np.newaxis, ...]

            # Generate the spectrogram
            spectrogram = PreProcessingPipelineBaseline.subtract_spectrum_s(fid_noise, t)

            # Pad zeros to the spectrogram
            spectrogram_padd = pad_zeros_spectrogram(spectrogram[0])
            spectrogram = spectrogram_padd[np.newaxis, ...]

            # Convert the spectrogram and ppm to Torch tensors
            spectrogram = torch.from_numpy(spectrogram)
            ppm = torch.from_numpy(ppm)

            # Create a three-channel spectrogram by concatenating the spectrogram with itself
            three_channels_spectrogram = torch.cat([spectrogram, spectrogram, spectrogram])

            # Convert the three-channel spectrogram to FloatTensor
            inputs = three_channels_spectrogram.real.type(torch.FloatTensor)
            inputs = inputs.to(device)

            # Add a batch dimension to the inputs
            inputs = torch.unsqueeze(inputs, dim=0)

            model = models[model_type].to(device)
            # Get the predicted labels from the model
            pred_labels = model(inputs)

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
