import numpy as np
import json
import h5py
import os

###########################################
######## TAAD Utility Functions ###########
###########################################

def training_precision_recall(preds, sharp_labels, conf_thresh=0.15, nms_window=15, delta=15):

    """
    preds: (C=9, B, M, T) probabilities, with class 0 = background
    sharp_labels: (B, M, T) integer labels, 0 = background
    conf_thresh: confidence threshold on class probabilities
    nms_window: window for local-maximum suppression
    delta: max frame difference to match a detection to a GT event

    Returns:
        precision, recall  (floats)
    """

    C, B, M, T = preds.shape

    ### Keep action of highest score for each time step

    cls_map = np.argmax(preds, axis=0) # (B,M,T)
    score_map = np.take_along_axis(preds, cls_map[None], axis=0)[0] # (B,M,T)

    ### Apply threshold to keep action that have a higher score, and set the others to background

    below = score_map < conf_thresh
    cls_map[below] = 0
    score_map[below] = 1.0

    ### Apply NMS per batch and player and per class (as a player can perform successive actions)

    for b in range(B):
        for m in range(M):
            for c in range(1, C):  # skip background
                t_idxs = np.where(cls_map[b, m] == c)[0]
                if t_idxs.size == 0:
                    continue
                # Sort by score desc
                scores = score_map[b, m, t_idxs]
                order = np.argsort(-scores)
                t_sorted = t_idxs[order]

                kept = []
                suppressed = np.zeros(t_sorted.shape[0], dtype=bool)

                for i, t0 in enumerate(t_sorted):
                    if suppressed[i]:
                        continue
                    kept.append(t0)
                    # suppress neighbors within +/- nms_window
                    lo = t0 - nms_window
                    hi = t0 + nms_window
                    # tag any others in this neighborhood
                    for j in range(i + 1, t_sorted.shape[0]):
                        if suppressed[j]:
                            continue
                        tj = t_sorted[j]
                        if lo <= tj <= hi:
                            suppressed[j] = True

                # Set suppressed ones to background
                suppressed_times = np.setdiff1d(t_idxs, np.array(kept, dtype=int), assume_unique=False)
                if suppressed_times.size > 0:
                    cls_map[b, m, suppressed_times] = 0
                    score_map[b, m, suppressed_times] = 1.0

    ### Collect GT events ###

    gt_events = {} # (b,m,c)
    total_gt = 0
    total_gt_per_class = np.zeros(C, dtype=int)
    for b in range(B):
        for m in range(M):
            lbl = sharp_labels[b, m]
            idxs = np.where(lbl != 0)[0]
            for t in idxs:
                c = int(lbl[t])
                gt_events.setdefault((b, m, c), []).append(int(t))
                total_gt += 1
                total_gt_per_class[c] += 1

    ### Collect Detections ###

    dets = [] # (score, b, m, c, t)
    for b in range(B):
        for m in range(M):
            t_idxs = np.where(cls_map[b, m] != 0)[0]
            for t in t_idxs:
                c = int(cls_map[b, m, t])
                s = float(score_map[b, m, t])
                dets.append((s, b, m, c, int(t)))

    dets.sort(key=lambda x: x[0], reverse=True)

    ### Greedy matching ###

    matched_gt = {k: np.zeros(len(v), dtype=bool) for k, v in gt_events.items()}
    TP = 0
    FP = 0
    TP_per_class = np.zeros(C, dtype=int)
    FP_per_class = np.zeros(C, dtype=int)

    for s, b, m, c, t_det in dets:
        key = (b, m, c)
        if key not in gt_events:
            FP += 1
            FP_per_class[c] += 1
            continue
        times = gt_events[key]
        used = matched_gt[key]
        best_i = -1
        best_dt = delta + 1
        for i, (t_gt, u) in enumerate(zip(times, used)):
            if u:
                continue
            dt = abs(t_gt - t_det)
            if dt <= delta and dt < best_dt:
                best_dt = dt
                best_i = i
        if best_i >= 0:
            used[best_i] = True
            TP += 1
            TP_per_class[c] += 1
        else:
            FP += 1
            FP_per_class[c] += 1

    FN = total_gt - TP
    FN_per_class = total_gt_per_class - TP_per_class

    ### per-class precision/recall

    precision_per_class = np.divide(TP_per_class, TP_per_class + FP_per_class, out=np.zeros_like(TP_per_class, dtype=float), where=(TP_per_class + FP_per_class) > 0)
    recall_per_class = np.divide(TP_per_class, TP_per_class + FN_per_class, out=np.zeros_like(TP_per_class, dtype=float), where=(TP_per_class + FN_per_class) > 0)

    precision = precision_per_class
    recall    = recall_per_class

    return precision, recall


def np_softmax(batch):

    return np.exp(batch.astype(np.float64)) / (np.exp(batch.astype(np.float64))).sum(0)



def post_processing_TAAD_preds_and_export(nms_window, root_path, save_path, h5file_path):


    class_names = ['background', 'drive', 'pass', 'cross', 'throw-in', 'shot', 'header', 'tackle', 'block']
    FRAME, PLAYER_ID, LEFT_TO_RIGHT, SHIRT_NUMBER, ROLE_ID, X_POS, Y_POS, X_SPEED, Y_SPEED, ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT, CLS = range(14)

    C = len(class_names)

    h5_file = h5py.File(h5file_path, 'r')
    keys = list(h5_file.keys())

    all_events = {"keys": [], "events": {}}

    for kidx, k in enumerate(keys):

        data = h5_file[k][:].astype(np.float64)

        with open(os.path.join(root_path, r"TAAD_predictions", f'avg_logits_{k}.npy'), 'rb') as f:
            preds = np.load(f).astype(np.float32)  # (9,M,T)
        preds = np_softmax(preds)

        _, M, T = preds.shape

        preds_cls = preds.argmax(0)  # (M,T)
        preds_scores = preds.max(0)  # (M,T)

        # NMS 
        for m in range(M):
            for c in range(1, C):  # skip background
                t_idxs = np.where(preds_cls[m] == c)[0]
                if t_idxs.size == 0:
                    continue
                scores = preds_scores[m, t_idxs]
                order = np.argsort(-scores)
                t_sorted = t_idxs[order]

                kept = []
                suppressed = np.zeros(t_sorted.shape[0], dtype=bool)

                for i, t0 in enumerate(t_sorted):
                    if suppressed[i]:
                        continue
                    kept.append(t0)
                    lo = t0 - nms_window
                    hi = t0 + nms_window
                    for j in range(i + 1, t_sorted.shape[0]):
                        if suppressed[j]:
                            continue
                        tj = t_sorted[j]
                        if lo <= tj <= hi:
                            suppressed[j] = True

                suppressed_times = np.setdiff1d(t_idxs, np.array(kept, dtype=int), assume_unique=False)
                if suppressed_times.size > 0:
                    preds_cls[m, suppressed_times] = 0
                    preds_scores[m, suppressed_times] = 1.0

        # Collect detections as (t m,c,score)
        raw_events = []  # (t,m,c,score)
        for m in range(M):
            t_idxs = np.where(preds_cls[m] != 0)[0]
            for t in t_idxs:
                c = int(preds_cls[m, t])
                s = float(preds_scores[m, t])
                raw_events.append((int(t), int(m), int(c), s))

        # raw_events.sort(key=lambda x: x[3], reverse=True)

        # Build per frame shirt lookup to handle substitutions or role changes
        PLAYERS_PER_TEAM = 13
        T_pred = T

        # Initialize with -1 (for unknown)
        shirt_by_time = np.full((M, T_pred), -1, dtype=int)

        # Use rows that have a known shirt number
        mask_sn = ~np.isnan(data[:, SHIRT_NUMBER])
        if np.any(mask_sn):
            frs  = data[mask_sn, FRAME].astype(int)
            ltrs = data[mask_sn, LEFT_TO_RIGHT].astype(int)
            rids = data[mask_sn, ROLE_ID].astype(int)
            shs  = data[mask_sn, SHIRT_NUMBER].astype(int)

            # keep only frames within [0, T-1] and valid role ids
            valid = (frs >= 0) & (frs < T_pred) & (rids >= 1) & (rids <= PLAYERS_PER_TEAM)
            frs, ltrs, rids, shs = frs[valid], ltrs[valid], rids[valid], shs[valid]

            ms = ltrs * PLAYERS_PER_TEAM + (rids - 1)
            # If multiple entries for the same (m, frame), the last seen wins
            for m_idx, fr, sn in zip(ms, frs, shs):
                shirt_by_time[m_idx, fr] = sn

        # Forward fill along time (last known shirt up to that frame)
        for m_idx in range(M):
            last = -1
            for tt in range(T_pred):
                if shirt_by_time[m_idx, tt] == -1:
                    shirt_by_time[m_idx, tt] = last
                else:
                    last = shirt_by_time[m_idx, tt]

        # Backward fill for unknowns at the start (use first future known shirt)
        for m_idx in range(M):
            nxt = -1
            for tt in range(T_pred - 1, -1, -1):
                if shirt_by_time[m_idx, tt] == -1:
                    shirt_by_time[m_idx, tt] = nxt
                else:
                    nxt = shirt_by_time[m_idx, tt]

        # Transform to (frame, team_left_right, shirt_number, class, score) for JSON export and greedy matching (for evaluation)
        events = []
        for t_det, m_idx, c_id, s in raw_events:
            team_left_right = m_idx // PLAYERS_PER_TEAM
            shirt_number = int(shirt_by_time[m_idx, t_det]) if (0 <= t_det < T_pred) else -1
            events.append([int(t_det), int(team_left_right), int(shirt_number), int(c_id), float(s)])
        
        all_events["keys"].append(k)
        all_events["events"][k] = events

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_events, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(all_events['keys'])} games, total events = {sum(len(v) for v in all_events['events'].values())}")
    print(f"JSON written to: {save_path}")


########################################################
######## DST Logits2Events Utility Functions ###########
########################################################


def calculate_precision_recall(predictions, labels, delta):
    """
    Calculate precision and recall based on predictions and labels.
    
    Parameters:
    - predictions: numpy array of shape (B, T), predicted class for each frame.
    - labels: numpy array of shape (B, T), true labels for each frame.
    - delta: int, temporal distance threshold for matching.
    
    Returns:
    - precision: float, overall precision across all batches.
    - recall: float, overall recall across all batches.
    """

    B, T = predictions.shape
    tp = 0 
    fp = 0
    fn = 0
    
    for b in range(B):
        pred_used = np.zeros(T, dtype=bool)  
        label_used = np.zeros(T, dtype=bool)
        
        # Match predictions with labels
        for t in range(T):

            pred_class = predictions[b, t]
            
            if pred_class != 0: 
                
                start = max(0, t - delta)
                end = min(T, t + delta + 1)
                
                match_idx = np.where((labels[b, start:end] == pred_class) & ~label_used[start:end])[0]
                if len(match_idx) > 0:
                    tp += 1
                    label_used[start + match_idx[0]] = True
                else:
                    fp += 1

        fn += np.sum((labels[b, :] != 0) & ~label_used) 
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return precision, recall


def build_role_sequences(preds, frames, labels, labframes):

    B,T = preds.shape

    max_frame = int(max(frames.max(), labframes.max()))

    predseq = np.zeros((B,max_frame+1))
    labelsseq = np.zeros((B,max_frame+1))

    for b in range(B):
        for t in range(T):
            if not (preds[b,t] in [26]) :
                curr_frame = int(frames[b,t])
                predseq[b,curr_frame] = preds[b,t] + 1

    B,T = labels.shape
    for b in range(B):
        for t in range(T):
            if not (labels[b,t] in [26]) :
                curr_frame = int(labframes[b,t])
                labelsseq[b,curr_frame] = labels[b,t] + 1
    
    return predseq, labelsseq


def build_action_sequences(preds, frames, labels, labframes):

    B,T = preds.shape

    max_frame = int(max(frames.max(), labframes.max()))

    predseq = np.zeros((B,max_frame+1))
    labelsseq = np.zeros((B,max_frame+1))

    for b in range(B):
        for t in range(T):
            if not (preds[b,t] in [0,9,10]) :
                curr_frame = int(frames[b,t])
                predseq[b,curr_frame] = preds[b,t]

    B,T = labels.shape
    for b in range(B):
        for t in range(T):
            if not (labels[b,t] in [0,9,10]) :
                curr_frame = int(labframes[b,t])
                labelsseq[b,curr_frame] = labels[b,t]
    
    return predseq, labelsseq


###############################################
######## Evaluation on full matches ###########
###############################################


def evaluate_events_from_json(
    gt_json_path,
    pred_json_path,
    class_names,
    delta=12,
    conf_thresh=0.15,
    print_per_game=False):
    """
    JSON structure expected (both GT and Pred):
      {
        "keys": [k1, k2, ...],
        "events": {
           k1: [[frame, team, shirt, class, ...], ...],
           k2: ...
        }
      }

    Ground truth events are expected as: [frame, team, shirt, class, bbox, replay]
      - bbox: 1 if a bbox exists for that event, else 0
      - replay: 1 if the frame has no visible players (all ROI_X NaN), else 0

    Predicted events are expected as: [frame, team, shirt, class, score]

    Matching:
      - Group events by (team, shirt, class).
      - For each prediction, pick the closest unused GT
        within +/- delta frames in the same group.
      - Counting TPs, FPs, and FNs follows a classic greedy strategy.

    Splits:
      - TP_* splits are based on the matched GT event's (bbox, replay).
      - FN_* splits are based on the unmatched GT events' (bbox, replay).

    Prints a per-class table and overall summary.
    Returns a dict with aggregated metrics and splits.
    """
    import json
    import numpy as np

    C = len(class_names)
    with open(gt_json_path, "r", encoding="utf-8") as f:
        gt = json.load(f)
    with open(pred_json_path, "r", encoding="utf-8") as f:
        pr = json.load(f)

    gt_keys = [str(k) for k in gt.get("keys", [])]
    pr_keys = [str(k) for k in pr.get("keys", [])]
    keys_to_eval = sorted(set(gt_keys).intersection(pr_keys))

    # Aggregators (overall)
    TP_per_class_all = np.zeros(C, dtype=int)
    FP_per_class_all = np.zeros(C, dtype=int)
    GT_per_class_all = np.zeros(C, dtype=int)

    # TP/FN splits (per class)
    TP_bb_per_class_all     = np.zeros(C, dtype=int)  # TP with bbox==1
    TP_nobb_per_class_all   = np.zeros(C, dtype=int)  # TP with bbox==0
    TP_rep_per_class_all    = np.zeros(C, dtype=int)  # TP with replay==1
    TP_live_per_class_all   = np.zeros(C, dtype=int)  # TP with replay==0

    FN_bb_per_class_all     = np.zeros(C, dtype=int)  # FN with bbox==1
    FN_nobb_per_class_all   = np.zeros(C, dtype=int)  # FN with bbox==0
    FN_rep_per_class_all    = np.zeros(C, dtype=int)  # FN with replay==1
    FN_live_per_class_all   = np.zeros(C, dtype=int)  # FN with replay==0

    total_TP_all = 0
    total_FP_all = 0
    total_GT_all = 0

    missing_gt = sorted(set(pr_keys) - set(gt_keys))
    missing_pr = sorted(set(gt_keys) - set(pr_keys))
    if missing_gt:
        print(f"[Info] {len(missing_gt)} predicted games not in GT (ignored): {missing_gt[:5]}{'...' if len(missing_gt)>5 else ''}")
    if missing_pr:
        print(f"[Info] {len(missing_pr)} GT games with no predictions (counted as all FN): {missing_pr[:5]}{'...' if len(missing_pr)>5 else ''}")

    for k in keys_to_eval:
        gt_list = gt["events"].get(k, [])
        pr_list = pr["events"].get(k, [])

        # Build GT buckets: (team, shirt, class) -> list of (t, bbox, replay)
        gt_events = {}
        total_gt_per_class = np.zeros(C, dtype=int)

        for e in gt_list:
            if len(e) < 4:
                continue
            t_gt   = int(e[0])
            team   = int(e[1])
            shirt  = int(e[2])
            c      = int(e[3])
            bbox   = int(e[4]) if len(e) > 4 else 0
            replay = int(e[5]) if len(e) > 5 else 0
            if c <= 0 or c >= C:
                continue
            key_grp = (team, shirt, c)
            gt_events.setdefault(key_grp, []).append((t_gt, bbox, replay))
            total_gt_per_class[c] += 1

        for key_grp in gt_events:
            gt_events[key_grp].sort(key=lambda x: x[0])

        # Collect predictions: (score, team, shirt, class, t)
        dets = []
        for e in pr_list:
            if len(e) < 4:
                continue
            t_det  = int(e[0])
            team   = int(e[1])
            shirt  = int(e[2])
            c      = int(e[3])
            score  = float(e[4]) if len(e) >= 5 else 1.0
            if score < conf_thresh:
                continue
            if c <= 0 or c >= C:
                continue
            dets.append((score, team, shirt, c, t_det))

        dets.sort(key=lambda x: x[0], reverse=True)

        # Greedy matching with splits
        matched_gt = {kk: np.zeros(len(v), dtype=bool) for kk, v in gt_events.items()}
        TP = 0
        FP = 0
        TP_per_class = np.zeros(C, dtype=int)
        FP_per_class = np.zeros(C, dtype=int)

        # Local per-class split counters
        TP_bb_per_class   = np.zeros(C, dtype=int)
        TP_nobb_per_class = np.zeros(C, dtype=int)
        TP_rep_per_class  = np.zeros(C, dtype=int)
        TP_live_per_class = np.zeros(C, dtype=int)

        for score, team, shirt, c, t_det in dets:
            key_grp = (team, shirt, c)
            if key_grp not in gt_events:
                FP += 1
                FP_per_class[c] += 1
                continue

            times = gt_events[key_grp] # list of (t_gt, bbox, replay)
            used  = matched_gt[key_grp]

            best_i = -1
            best_dt = delta + 1
            for i, ((t_gt, bbox, replay), u) in enumerate(zip(times, used)):
                if u:
                    continue
                dt = abs(t_gt - t_det)
                if dt <= delta and dt < best_dt:
                    best_dt = dt
                    best_i = i

            if best_i >= 0:
                used[best_i] = True
                TP += 1
                TP_per_class[c] += 1

                _, bbox_g, replay_g = times[best_i]
                if bbox_g == 1: TP_bb_per_class[c]   += 1
                else:           TP_nobb_per_class[c] += 1
                if replay_g == 1: TP_rep_per_class[c]  += 1
                else:             TP_live_per_class[c] += 1
            else:
                FP += 1
                FP_per_class[c] += 1

        total_gt = int(sum(total_gt_per_class))
        FN = total_gt - TP
        FN_per_class = total_gt_per_class - TP_per_class

        # Compute FN splits from unmatched GT
        FN_bb_per_class   = np.zeros(C, dtype=int)
        FN_nobb_per_class = np.zeros(C, dtype=int)
        FN_rep_per_class  = np.zeros(C, dtype=int)
        FN_live_per_class = np.zeros(C, dtype=int)

        for key_grp, times in gt_events.items():
            used = matched_gt[key_grp]
            # times[i] = (t_gt, bbox, replay)
            for (t_gt, bbox_g, replay_g), u in zip(times, used):
                if not u:
                    # unmatched -> FN
                    c = key_grp[2]
                    if bbox_g == 1: FN_bb_per_class[c]   += 1
                    else:           FN_nobb_per_class[c] += 1
                    if replay_g == 1: FN_rep_per_class[c]  += 1
                    else:             FN_live_per_class[c] += 1

        # Aggregate across games
        TP_per_class_all += TP_per_class
        FP_per_class_all += FP_per_class
        GT_per_class_all += total_gt_per_class

        TP_bb_per_class_all   += TP_bb_per_class
        TP_nobb_per_class_all += TP_nobb_per_class
        TP_rep_per_class_all  += TP_rep_per_class
        TP_live_per_class_all += TP_live_per_class

        FN_bb_per_class_all   += FN_bb_per_class
        FN_nobb_per_class_all += FN_nobb_per_class
        FN_rep_per_class_all  += FN_rep_per_class
        FN_live_per_class_all += FN_live_per_class

        total_TP_all += TP
        total_FP_all += FP
        total_GT_all += total_gt

        if print_per_game:
            PR_g  = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            REC_g = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            F1_g  = (2 * PR_g * REC_g / (PR_g + REC_g)) if (PR_g + REC_g) > 0 else 0.0
            print(f'GAME {k}: Overall PR={PR_g:.3f}  REC={REC_g:.3f}  F1={F1_g:.3f}')

    # Final aggregated report
    FN_per_class_all = GT_per_class_all - TP_per_class_all

    precision_per_class_all = np.divide(
        TP_per_class_all, TP_per_class_all + FP_per_class_all,
        out=np.zeros_like(TP_per_class_all, dtype=float),
        where=(TP_per_class_all + FP_per_class_all) > 0
    )
    recall_per_class_all = np.divide(
        TP_per_class_all, TP_per_class_all + FN_per_class_all,
        out=np.zeros_like(TP_per_class_all, dtype=float),
        where=(TP_per_class_all + FN_per_class_all) > 0
    )

    overall_PR  = total_TP_all / (total_TP_all + total_FP_all) if (total_TP_all + total_FP_all) > 0 else 0.0
    overall_REC = total_TP_all / total_GT_all if total_GT_all > 0 else 0.0
    overall_F1  = (2 * overall_PR * overall_REC / (overall_PR + overall_REC)) if (overall_PR + overall_REC) > 0 else 0.0

    # Compute GT splits from TP/FN splits
    GT_bb_per_class_all   = TP_bb_per_class_all   + FN_bb_per_class_all
    GT_nobb_per_class_all = TP_nobb_per_class_all + FN_nobb_per_class_all
    GT_rep_per_class_all  = TP_rep_per_class_all  + FN_rep_per_class_all
    GT_live_per_class_all = TP_live_per_class_all + FN_live_per_class_all

    # Totals for split columns
    tot_tp_bb   = int(TP_bb_per_class_all.sum())
    tot_tp_nobb = int(TP_nobb_per_class_all.sum())
    tot_tp_rep  = int(TP_rep_per_class_all.sum())
    tot_tp_live = int(TP_live_per_class_all.sum())

    tot_fn_bb   = int(FN_bb_per_class_all.sum())
    tot_fn_nobb = int(FN_nobb_per_class_all.sum())
    tot_fn_rep  = int(FN_rep_per_class_all.sum())
    tot_fn_live = int(FN_live_per_class_all.sum())

    tot_gt_bb   = int(GT_bb_per_class_all.sum())
    tot_gt_nobb = int(GT_nobb_per_class_all.sum())
    tot_gt_rep  = int(GT_rep_per_class_all.sum())
    tot_gt_live = int(GT_live_per_class_all.sum())

    # Print results
    name_col_w = max(len("Class"), max(len(n) for n in class_names[1:]))
    hdr = (
        f"{'Class'.ljust(name_col_w)}  "
        f"{'Precision':>9}  {'Recall':>9}  "
        f"{'TP':>6}  {'FP':>6}  {'FN':>6}  {'GT':>6}  "
        f"{'TP_bb':>6}  {'TP_noBB':>8}  {'TP_rep':>7}  {'TP_live':>8}  "
        f"{'FN_bb':>6}  {'FN_noBB':>8}  {'FN_rep':>7}  {'FN_live':>8}  "
        f"{'GT_bb':>6}  {'GT_noBB':>8}  {'GT_rep':>7}  {'GT_live':>8}"
    )
    sep = "-" * len(hdr)

    print("\n=== Aggregated Precision/Recall per Class (ALL GAMES) ===")
    print(hdr)
    print(sep)
    for c in range(1, C):
        name = class_names[c].ljust(name_col_w)
        prc  = f"{precision_per_class_all[c]:.3f}".rjust(9)
        rcl  = f"{recall_per_class_all[c]:.3f}".rjust(9)
        tp   = f"{TP_per_class_all[c]}".rjust(6)
        fp   = f"{FP_per_class_all[c]}".rjust(6)
        fn   = f"{FN_per_class_all[c]}".rjust(6)
        gt   = f"{GT_per_class_all[c]}".rjust(6)

        tp_bb   = f"{TP_bb_per_class_all[c]}".rjust(6)
        tp_nobb = f"{TP_nobb_per_class_all[c]}".rjust(8)
        tp_rep  = f"{TP_rep_per_class_all[c]}".rjust(7)
        tp_live = f"{TP_live_per_class_all[c]}".rjust(8)

        fn_bb   = f"{FN_bb_per_class_all[c]}".rjust(6)
        fn_nobb = f"{FN_nobb_per_class_all[c]}".rjust(8)
        fn_rep  = f"{FN_rep_per_class_all[c]}".rjust(7)
        fn_live = f"{FN_live_per_class_all[c]}".rjust(8)

        gt_bb   = f"{GT_bb_per_class_all[c]}".rjust(6)
        gt_nobb = f"{GT_nobb_per_class_all[c]}".rjust(8)
        gt_rep  = f"{GT_rep_per_class_all[c]}".rjust(7)
        gt_live = f"{GT_live_per_class_all[c]}".rjust(8)

        print(
            f"{name}  {prc}  {rcl}  {tp}  {fp}  {fn}  {gt}  "
            f"{tp_bb}  {tp_nobb}  {tp_rep}  {tp_live}  "
            f"{fn_bb}  {fn_nobb}  {fn_rep}  {fn_live}  "
            f"{gt_bb}  {gt_nobb}  {gt_rep}  {gt_live}"
        )

    print(sep)
    print(
        f"{'OVERALL'.ljust(name_col_w)}  "
        f"{overall_PR:>9.3f}  {overall_REC:>9.3f}  "
        f"{total_TP_all:>6d}  {total_FP_all:>6d}  {(total_GT_all - total_TP_all):>6d}  {total_GT_all:>6d}  "
        f"{tot_tp_bb:>6d}  {tot_tp_nobb:>8d}  {tot_tp_rep:>7d}  {tot_tp_live:>8d}  "
        f"{tot_fn_bb:>6d}  {tot_fn_nobb:>8d}  {tot_fn_rep:>7d}  {tot_fn_live:>8d}  "
        f"{tot_gt_bb:>6d}  {tot_gt_nobb:>8d}  {tot_gt_rep:>7d}  {tot_gt_live:>8d}"
    )
    print(f"Micro F1: {overall_F1:.3f}")

    return {
        "TP_per_class": TP_per_class_all,
        "FP_per_class": FP_per_class_all,
        "FN_per_class": FN_per_class_all,
        "GT_per_class": GT_per_class_all,
        "precision_per_class": precision_per_class_all,
        "recall_per_class": recall_per_class_all,
        "overall": {
            "TP": total_TP_all,
            "FP": total_FP_all,
            "GT": total_GT_all,
            "precision": overall_PR,
            "recall": overall_REC,
            "f1": overall_F1,
        },
        "tp_splits": {
            "bbox": TP_bb_per_class_all,
            "no_bbox": TP_nobb_per_class_all,
            "replay": TP_rep_per_class_all,
            "live": TP_live_per_class_all,
        },
        "fn_splits": {
            "bbox": FN_bb_per_class_all,
            "no_bbox": FN_nobb_per_class_all,
            "replay": FN_rep_per_class_all,
            "live": FN_live_per_class_all,
        },
        "gt_splits": {
            "bbox": GT_bb_per_class_all,
            "no_bbox": GT_nobb_per_class_all,
            "replay": GT_rep_per_class_all,
            "live": GT_live_per_class_all,
        },
        "keys_evaluated": keys_to_eval,
    }

