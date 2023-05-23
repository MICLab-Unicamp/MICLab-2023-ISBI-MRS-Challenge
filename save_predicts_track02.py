import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
from utils import ReadDatasets, pad_zeros_spectrogram, read_yaml
from pre_processing import PreProcessingPipelineBaseline
from models import TimmSimpleCNN

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Predicted neural network MRS")
    parser.add_argument("config_file", type=str, help="Config neural network YAML")
    parser.add_argument("weights", type=str, help="Weights neural network")
    parser.add_argument("test_data_path", type=str, help="Path to test dataset (.h5)")
    parser.add_argument("save_folder_path", type=str, help="add folder path which the predict .h5 file will be saved")
    args = parser.parse_args()

    # Determine device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read configuration YAML file
    configs = read_yaml(args.config_file)

    # Initialize the model
    model = TimmSimpleCNN(**configs["model"]["TimmSimpleCNN"]).to(device)

    # Load pre-trained weights
    load_dict = torch.load(args.weights)
    model.load_state_dict(load_dict["model_state_dict"])

    # Load test data
    test_data_path = args.test_data_path
    input_transients, input_ppm, input_t = ReadDatasets.read_h5_sample_track_2(test_data_path)

    pred_labels_stacked = None
    ppm_stacked = None

    # Process each input sample
    for i in tqdm(range(input_transients.shape[0])):
        fid_noise = input_transients[i, :, :, :]
        ppm = input_ppm[i, :]
        t = input_t[i, :]

        # Pre-processing steps
        fid_noise = fid_noise[np.newaxis, ...]
        spectrogram = PreProcessingPipelineBaseline.subtract_spectrum_s(fid_noise, t)
        spectrogram_padd = pad_zeros_spectrogram(spectrogram[0])
        spectrogram = spectrogram_padd[np.newaxis, ...]

        # Convert numpy arrays to torch tensors
        spectrogram = torch.from_numpy(spectrogram)
        ppm = torch.from_numpy(ppm)

        # Create three-channel spectrogram
        three_channels_spectrogram = torch.cat([spectrogram, spectrogram, spectrogram])

        # Prepare input data for the model
        inputs = three_channels_spectrogram.real.type(torch.FloatTensor).to(device)
        inputs = torch.unsqueeze(inputs, dim=0)

        # Forward pass through the model
        pred_labels = model(inputs).detach().cpu().numpy()
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
