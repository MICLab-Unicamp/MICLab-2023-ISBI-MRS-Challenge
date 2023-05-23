# Edited_MRS_challenge_Deep_Spectral_Divers_Team

This repository contains the code for the Deep Spectral Divers team's **first-place solution** to the **Edited-MRS Reconstruction Challenge** presented at the **IEEE International Symposium on Biomedical Imaging in 2023**. It enables inference using the models used in the challenge, aiming to reconstruct GABA spectra faster by utilizing less data compared to current Edited-MRS scans. For each track, we trained a separate model, resulting in four final models: Track 01, Track 02, and two instances of Track 03, one with 2048 data points and the other with 4096 data points.

Our team was the overall winner of the challenge.

## Challenge Description

Edited Magnetic Resonance Spectroscopy (MRS) is used to quantify metabolites that are overlapped by those of higher concentration in typical scans. The challenge focused on quantifying Gamma Aminobutyric Acid (GABA), which is overlapped by creatine and glutamate. High-quality data in Edited-MRS scans requires long scan times. The challenge proposed the use of machine learning models to reconstruct spectra using three quarters less data, enabling four times faster Edited-MRS scans.

Participants in the challenge were provided with simulated and in vivo data training sets representing GABA-edited MEGA-PRESS scans composed of two subspectra (ON and OFF). Scripts for data augmentation, including adding noise, frequency, and phase shifts, were also provided. The models submitted by the teams were evaluated on simulated data (Track 01), homogeneous in vivo data (single-vendor) (Track 02), and heterogeneous in vivo data (multi-vendor) (Track 03) using quantitative metrics such as mean squared error, signal-to-noise ratio, linewidth, and peak shape.

The results of the challenge were presented at the IEEE International Symposium on Biomedical Imaging (ISBI) conference held in Cartagena, Colombia on April 18th, 2023. The challenge outcomes were summarized and submitted for a joint publication.

For more details about the challenge, you can visit the [challenge webpage](https://sites.google.com/view/edited-mrs-rec-challenge/home?authuser=0).


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/MICLab-Unicamp/MICLab-2023-ISBI-MRS-Challenge.git

2. Navigate to the project directory:

   ```bash
   cd MICLab-2023-ISBI-MRS-Challenge
   
3. Check the Python version in requirements.txt and install the required dependencies:

    ```bash
   pip install -r requirements.txt
   
## Usage

Getting the predictions from the models:

1. Ensure you have the test data for each one of the tracks in the right .h5 format.

To execute the model's inference, you must have an h5 file containing the correctly formatted test data for each track (according to the challenge). The h5 file should be structured with the following datasets to ensure compatibility:

**For Track 1:**

- Dataset Name: `"ppm"`
  - Dataset Type: 1D array (`np.ndarray`)
  - Contains the ppm values.

- Dataset Name: `"t"`
  - Dataset Type: 1D array (`np.ndarray`)
  - Contains the time values.

- Dataset Name: `"transients"`
  - Dataset Type: 1D array (`np.ndarray`)
  - Contains the transients data.

**For Track 2:**

- Dataset Name: `"ppm"`
  - Dataset Type: 1D array (`np.ndarray`)
  - Contains the ppm values.

- Dataset Name: `"t"`
  - Dataset Type: 1D array (`np.ndarray`)
  - Contains the time values.

- Dataset Name: `"transient_fids"`
  - Dataset Type: 1D array (`np.ndarray`)
  - Contains the transients data.

**For Track 3:**

- Group Name: `"data_2048"`
  - Contains datasets for downsampled data.

  - Dataset Name: `"ppm"`
    - Dataset Type: 1D array (`np.ndarray`)
    - Contains the downsampled ppm values.

  - Dataset Name: `"t"`
    - Dataset Type: 1D array (`np.ndarray`)
    - Contains the downsampled time values.

  - Dataset Name: `"transient_fids"`
    - Dataset Type: 1D array (`np.ndarray`)
    - Contains the downsampled transients data.

- Group Name: `"data_4096"`
  - Contains datasets for upsampled data.

  - Dataset Name: `"ppm"`
    - Dataset Type: 1D array (`np.ndarray`)
    - Contains the upsampled ppm values.

  - Dataset Name: `"t"`
    - Dataset Type: 1D array (`np.ndarray`)
    - Contains the upsampled time values.

  - Dataset Name: `"transient_fids"`
    - Dataset Type: 1D array (`np.ndarray`)
    - Contains the upsampled transients data.

Make sure that the datasets are stored with the exact names and types mentioned above. The functions rely on these specific dataset names to extract the required data.

2. Get the trained weights from each model at this [Link](https://drive.google.com/drive/folders/1NJ1OGs-W9GZE9XMHvctKs67kjNOtY74O?usp=share_link).

3. Run the script:

Instructions for **Tracks 01 and 02**:

Execution:

    python3 save_predicts_track01.py [config_file] [weights] [test_data_path] [save_folder_path]

or 

    python3 save_predicts_track02.py [config_file] [weights] [test_data_path] [save_folder_path]

Example usage:

    python3 save_predicts_track02.py configs/config_track02.yaml weights/weights_track02.pt data/challenge_data/track_02_test_data.h5 data/save_predicts

Replace `[config_file]` with the path to the YAML configuration for the track.

Replace `[weights]` with the path to the weights file for the track.

Replace `[test_data_path]` with the path to the track .h5 file containing the test dataset.

Replace `[save_folder_path]` with the folder path which the predict .h5 file will be saved.

Instructions for **Track 03**:

Execution:

    python3 save_predicts_track03.py [config_file_down] [config_file_up] [weights_down] [weights_up] [test_data_path] [save_folder_path]

Example usage:

    python3 save_predicts_track03.py configs/config_track03_2048.yaml configs/config_track03_4096.yaml weights/weights_track03_2048.pt weights/weights_track03_4096.pt data/challenge_data/track_03_test_data.h5 data/save_predicts



Replace `[config_file_down]` with the path to the YAML configuration for the track 03 downsampled (2048).

Replace `[config_file_up]` with the path to the YAML configuration for the track 03 upsampled (4096).

Replace `[weights_down]` with the path to the weights file for the track 03 downsampled (2048).

Replace `[weights_up]` with the path to the weights file for the track 03 upsampled (4096).

Replace `[test_data_path]` with the path to the track .h5 file containing the test dataset.

Replace `[save_folder_path]` with the folder path which the predict .h5 file will be saved.

4. The script will perform inference on each sample in the test dataset using the model.

5. The predicted spectra and ppm values will be saved in an output file named track01.h5, track02.h5, or track03.h5, depending on the respective script. These files will be located in the folder `[save_folder_path]` provided.


## Citation

If you use our model inference in your research please cite

      @misc{MICLab-2023-ISBI-MRS-Challenge,
        author = {Dias, G. and Ueda, L. and Oliveira, M. and Dertkigil, S. and Costa, P. and Rittner, L.},
        title = {{Deep Spectral Divers first-place solution}},
        year = {2023},
        howpublished = {Code repository for the Deep Spectral Divers team's first-place solution to the Edited-MRS Reconstruction Challenge},
        url = {https://github.com/MICLab-Unicamp/MICLab-2023-ISBI-MRS-Challenge}
       }


 
