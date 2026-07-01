"""Utilities for loading MWT .mat files."""

import os
import pickle
from typing import List, Tuple

import numpy as np
import scipy.io


def reading_raw_data(path: str) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Read all .mat files from the MWT dataset folder.

    The MWT files used in this work store signals and labels in the MATLAB
    `Data` structure. The index positions below follow the structure used in
    the original experiment code.
    """
    mat_files = sorted([file for file in os.listdir(path) if file.endswith(".mat")])
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in: {path}")

    eeg_01_data, eeg_02_data, labels_01, labels_02 = [], [], [], []

    for mat_file in mat_files:
        file_path = os.path.join(path, mat_file)
        data_dict = scipy.io.loadmat(file_path)
        data = data_dict["Data"][0][0]

        eeg_01_data.append(data[1])
        eeg_02_data.append(data[2])
        labels_01.append(data[5])
        labels_02.append(data[6])

    return eeg_01_data, eeg_02_data, labels_01, labels_02


def load_data(path: str, cache_path: str = "eeg_data.pkl", read_raw: bool = False):
    """Load MWT EEG data from cache or raw .mat files."""
    if read_raw or not os.path.exists(cache_path):
        eeg_01_data, eeg_02_data, labels_01, labels_02 = reading_raw_data(path)
        with open(cache_path, "wb") as file:
            pickle.dump((eeg_01_data, eeg_02_data, labels_01, labels_02), file)
    else:
        with open(cache_path, "rb") as file:
            eeg_01_data, eeg_02_data, labels_01, labels_02 = pickle.load(file)

    return eeg_01_data, eeg_02_data, labels_01, labels_02
