import os
import argparse
from utils.metric_utils import evaluate_events_from_json


def parse_args():

    p = argparse.ArgumentParser(description="COMPUTE PRECISION, RECALL, F1 FROM JSON FILES")

    default_data_root = os.getcwd()
    default_pred_file = os.path.join(default_data_root, "playbyplay_PRED", "playbyplay_TAAD_val.json")
    default_gt_file = os.path.join(default_data_root, "playbyplay_GT", "playbyplay_val.json")

    p.add_argument("--predictions_file", type=str, default=default_pred_file, help="JSON file containing the predictions")
    p.add_argument("--ground_truth_file", type=str, default=default_gt_file, help="JSON file containing the ground truth")
    p.add_argument("--delta", type=int, default=12, help="Tolerance of + or - delta frames for the matching")
    p.add_argument("--confidence_threshold", type=float, default=0.15, help="Confidence threshold to accept or reject a prediction based on its score")
    p.add_argument("--print_per_game", type=bool, default=False, help="Print Precision and Recall per gampe from the set")

    return p.parse_args()

if __name__ == '__main__':

    args = parse_args()

    PRED_FILE = args.predictions_file
    GT_FILE = args.ground_truth_file
    DELTA = args.delta
    CONF_THRESH = args.confidence_threshold
    PRINT_PER_GAME = args.print_per_game

    class_names = ['background', 'drive', 'pass', 'cross', 'throw-in', 'shot', 'header', 'tackle', 'block']

    metrics = evaluate_events_from_json(
        gt_json_path=GT_FILE,
        pred_json_path=PRED_FILE,
        class_names=class_names,
        delta=DELTA,       
        conf_thresh=CONF_THRESH,
        print_per_game=PRINT_PER_GAME)