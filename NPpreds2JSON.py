import argparse
import os

from utils.metric_utils import post_processing_TAAD_preds_and_export

def parse_args():

    p = argparse.ArgumentParser(description="Post-Process and Compile TAAD predictions stored as .npy files into a JSON")

    default_data_root = os.getcwd()
    default_prediction_save_path = os.path.join(default_data_root, "TAAD_predictions")
    default_prediction_save_path_json = os.path.join(default_data_root, "playbyplay_PRED")

    p.add_argument("--data_root", type=str, default=default_data_root, help="Root folder with data/")
    p.add_argument("--NP_preds_path", type=str, default=default_prediction_save_path, help="Numpy TAAD predictions folder")
    p.add_argument("--save_path_json", type=str, default=default_prediction_save_path_json, help="Folder to write JSON predictions for evaluation")
    p.add_argument("--set_to_run", type=str, default=r"val", help="Part of the dataset to run : train, val, challenge")
    p.add_argument("--nms_window", type=int, default=25)

    return p.parse_args()


if __name__ == '__main__':

    args = parse_args()

    DATA_ROOT = args.data_root
    PRED_PATH = args.NP_preds_path
    SAVE_PATH = args.save_path_json
    SET_STATUS = args.set_to_run
    NMS_WINDOW = args.nms_window

    os.makedirs(SAVE_PATH, exist_ok=True)

    ##### POST PROCESS AND EXPORT AS JSON #####
    json_path = os.path.join(SAVE_PATH, f'playbyplay_TAAD_{SET_STATUS}.json')
    h5file_path = os.path.join(DATA_ROOT, "data", f'{SET_STATUS}_tactical_data.h5')
    post_processing_TAAD_preds_and_export(nms_window=NMS_WINDOW, root_path=DATA_ROOT, save_path=json_path, h5file_path=h5file_path)
