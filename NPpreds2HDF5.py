import argparse
import numpy as np
import os
import h5py
from tqdm import tqdm

def parse_args():

    p = argparse.ArgumentParser(description="Compile TAAD predictions stored as .npy files into one HDF5 file for DST training")

    default_data_root = os.getcwd()
    default_prediction_save_path = os.path.join(default_data_root, "TAAD_predictions")
    default_prediction_save_path_h5 = os.path.join(default_data_root, "TAAD_h5")

    p.add_argument("--data_root", type=str, default=default_data_root, help="Root folder with data/")
    p.add_argument("--NP_preds_path", type=str, default=default_prediction_save_path, help="Numpy TAAD predictions folder")
    p.add_argument("--save_path_h5", type=str, default=default_prediction_save_path_h5, help="Folder to write HDF5 TAAD predictions")
    p.add_argument("--set_to_run", type=str, default=r"train", help="Part of the dataset to run : train, val, challenge")

    return p.parse_args()


if __name__ == '__main__':

    args = parse_args()

    DATA_ROOT = args.data_root
    PRED_PATH = args.NP_preds_path
    SAVE_PATH_H5 = args.save_path_h5
    SET_STATUS = args.set_to_run

    os.makedirs(SAVE_PATH_H5, exist_ok=True)

    #### CONVERT NUMPY PREDICTIONS IN A UNIQUE HDF5 FILE FOR TRAINING OF DST #####

    h5file_path = os.path.join(DATA_ROOT, "data", f'{SET_STATUS}_tactical_data.h5')
    with h5py.File(h5file_path, 'r') as f:
        list_halftimes = list(f.keys())

    SAVE_FILENAME = os.path.join(SAVE_PATH_H5, f'preds_{SET_STATUS}.h5')

    with h5py.File(SAVE_FILENAME, "w") as h5f:
        for k in tqdm(list_halftimes) :
            path = os.path.join(PRED_PATH, f'avg_logits_{k}.npy')
            data = np.load(path)  
            h5f.create_dataset(name=k,data=data)
    
    print(f'File {SAVE_FILENAME} saved')
