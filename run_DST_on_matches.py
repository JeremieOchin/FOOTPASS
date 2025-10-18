import argparse
import numpy as np
import math
import os
import h5py
import json
import torch

from models.model_DST import DST_Logits2Events

def parse_args():

    p = argparse.ArgumentParser(description="run DST Baseline and store predictions")

    default_data_root = os.getcwd()

    p.add_argument("--model_checkpoint", type=str, default=r"curr_model_6.pt", help="Filename of the checkpoint")
    p.add_argument("--data_root", type=str, default=default_data_root, help="Root folder with data/")
    p.add_argument("--set_to_run", type=str, default=r"val", help="Part of the dataset to run : train, val, test")
    p.add_argument("--start_index", type=int, default=0, help="Index of the first hafltime to run in the list of hdf5 file")
    p.add_argument("--end_index", type=int, default=6, help="Index of the last hafltime to run in the list of hdf5 file")
    p.add_argument("--batch_size", type=int, default=40)

    p.add_argument("--framespan", type=int, default=750)
    p.add_argument("--inputdim", type=int, default=364)
    p.add_argument("--outputdim", type=int, default=37)
    p.add_argument("--hiddendim", type=int, default=512)
    p.add_argument("--numlayers", type=int, default=6)
    p.add_argument("--numheads", type=int, default=8)

    return p.parse_args()


if __name__ == '__main__':

    args = parse_args()

    DATA_ROOT = args.data_root
    MODEL_CHECKPOINT = args.model_checkpoint
    MODEL_CHECKPOINT = os.path.join(DATA_ROOT, "runs", "DST_Baseline", MODEL_CHECKPOINT)
    SAVE_PATH = args.save_path
    SET_STATUS = args.set_to_run
    START_IDX = args.start_index
    END_IDX = args.end_index

    BATCH_SIZE = args.batch_size

    FRAMESPAN = args.framespan
    INPUT_DIM = args.inputdim
    OUTPUT_DIM = args.outputdim
    HIDDEN_DIM = args.hiddendim
    NUM_LAYERS = args.numlayers
    NUM_HEADS = args.numheads
    DROPOUT = 0.1

    args_file = os.path.join(SAVE_PATH, "run_config.txt")
    with open(args_file, "w") as f:
        f.write("Inference configuration:\n")
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
    print(f"[INFO] Saved Inference configuration to {args_file}")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(device)


    def build_teamvector(data, logits):

        FRAME, PLAYER_ID, LEFT_TO_RIGHT, SHIRT_NUMBER, ROLE_ID, X_POS, Y_POS, X_SPEED, Y_SPEED, ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT, CLS = range(14) 

        ##############################
        #### LOAD DATA AND LOGITS ####

        data_min = int(data[:, FRAME].min())
        data_max = int(data[:, FRAME].max())
        logits_T = int(logits.shape[2])

        frame_range = np.arange(data_min, min(data_max, logits_T), dtype=np.int32)

        data = data[(data[:,FRAME]>=frame_range[0])&(data[:,FRAME]<=frame_range[-1])]
        logits = logits[:,:,frame_range[0]:(frame_range[-1] + 1)] # (9,M,T)
        action_data = data[data[:,CLS]!=0][:,[FRAME, LEFT_TO_RIGHT, ROLE_ID, CLS]].copy() # (T', 4) 'frame','left_to_right','short_role_id','class_id'

        ###########################
        #### BUILD TEAM VECTOR ####

        T = len(frame_range)
        M = 26

        # Normalize frame indices
        frame_raw = data[:, FRAME].astype(np.int32)
        frame0 = frame_range[0]
        frame_idx = frame_raw - frame0

        # Compute slot index
        role_idx = data[:, ROLE_ID].astype(np.int32) - 1
        slot_idx = data[:, LEFT_TO_RIGHT].astype(np.int32) * 13 + role_idx

        msk = ((frame_idx>=0)&(frame_idx<T)&(role_idx>=0)&(role_idx<13)&(slot_idx>=0)&(slot_idx<M))

        # Extract values
        x = data[msk, X_POS].astype(np.float32)
        y = data[msk, Y_POS].astype(np.float32)
        vx = data[msk, X_SPEED].astype(np.float32)
        vy = data[msk, Y_SPEED].astype(np.float32)
        roi_x = data[msk, ROI_X]
        obs = (~np.isnan(roi_x)).astype(np.float32) 
        fi = frame_idx[msk]
        si = slot_idx[msk]
        
        # Assign values to team vector array
        teamvec = np.full((5, M, T), -15.0, dtype=np.float32)

        teamvec[0, si, fi] = x
        teamvec[1, si, fi] = y
        teamvec[2, si, fi] = vx
        teamvec[3, si, fi] = vy
        teamvec[4, si, fi] = obs

        ################################################
        #### BUILD ENCODER INPUT AND DECODER TARGET ####

        #### ENCODER INPUT ####

        teamvec = np.ascontiguousarray(np.transpose(teamvec, (2, 0, 1)).reshape(T, -1)) # (T,130)
        logits = np.ascontiguousarray(np.transpose(logits, (2, 0, 1)).reshape(T, -1)) # (T,234)

        EncoderData = torch.from_numpy(np.concatenate([frame_range[:,None], teamvec, logits], axis=1)) # (T, 364)

        return EncoderData


    def one_hot_encode_with_padding(framesnb, N):

        B, T = framesnb.shape

        one_hot = torch.zeros(B, T, N + 2, dtype=torch.float32)  # N+2 to include the padding index
        tensor = framesnb.long()
        
        # Set one-hot encoding for valid frame numbers (0 to N)
        valid_mask = (tensor >= 0) & (tensor <= N)
        one_hot[torch.arange(B).unsqueeze(1), torch.arange(T).unsqueeze(0), tensor] = valid_mask.float()
        
        # Set one-hot encoding for padding (-1 -> index N+1)
        padding_mask = tensor == -1
        one_hot[torch.arange(B).unsqueeze(1), torch.arange(T).unsqueeze(0), torch.full_like(tensor, N + 1)] = padding_mask.float()
        
        return one_hot

    ##### INSTANTIATE MODEL #####

    model = DST_Logits2Events(framespan=FRAMESPAN,
                            input_dim=INPUT_DIM,
                            output_dim=OUTPUT_DIM,
                            hidden_dim=HIDDEN_DIM,
                            n_heads=NUM_HEADS,
                            n_enc_layers=NUM_LAYERS,
                            n_dec_layers=NUM_LAYERS,
                            dropout=DROPOUT)

    ckpt = torch.load(MODEL_CHECKPOINT, map_location=torch.device('cpu'))
    state_dict = ckpt['model_state_dict'] 
    model.load_state_dict(state_dict, strict=True)

    model = model.to(device)
    model.eval()

    #### MAKE PREDICTIONS ON FULL HALFTIMES #####

    logits_h5_path = os.path.join(DATA_ROOT, "TAAD_h5", f'preds_{SET_STATUS}.h5')
    data_h5_path = os.path.join(DATA_ROOT, "data", f'{SET_STATUS}_tactical_data.h5')

    FRAME, PLAYER_ID, LEFT_TO_RIGHT, SHIRT_NUMBER, ROLE_ID, X_POS, Y_POS, X_SPEED, Y_SPEED, ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT, CLS = range(14)
    PLAYERS_PER_TEAM = 13
    M= 26

    with h5py.File(data_h5_path, 'r') as data_h5_file :
        all_keys = list(data_h5_file.keys())

    all_events = {"keys": [], "events": {}}

    for k in all_keys :

        with h5py.File(data_h5_path, 'r') as data_h5_file, h5py.File(logits_h5_path, 'r') as logits_h5_file:
            datah5 = data_h5_file[k][:].astype(np.float64)
            logits = logits_h5_file[k][:].astype(np.float32)

        data = build_teamvector(datah5, logits)

        start_frame, end_frame = int(data[:,0].min()), int(data[:,0].max())
        frame_range = np.arange(start_frame, end_frame + 1)
        T = frame_range.shape[0]

        ### PAD AND PREP DATA
        pad_length = 750*math.ceil(data.shape[0]/750) - data.shape[0]
        data = np.concatenate([data, np.zeros((pad_length,365))], axis=0)

        data = data.reshape(-1,750,365)
        init_frames = data[:,0,0][:, None].copy()
        data[:,:,0] -= (init_frames - 1)
        enc_abs_framenb = data[:,:,0].copy()
        enc_abs_framenb[enc_abs_framenb<0] = -1
        data = data[:,:,1:].copy()

        data = torch.from_numpy(data)
        enc_abs_framenb = torch.from_numpy(enc_abs_framenb)
        src_key_padding_mask = (enc_abs_framenb == -1) 
        src = torch.cat([data, one_hot_encode_with_padding(enc_abs_framenb,750)], dim=-1).float()

        with torch.no_grad() :
            preds = model.forward_autoregressive(src=src.cuda(), src_frames=enc_abs_framenb.cuda(), src_key_padding_mask=src_key_padding_mask.cuda(), max_tgt_len=25)

        action_preds = preds[0].argmax(-1).cpu().numpy() # (B,max_length) Dim 10
        score_preds = preds[0].max(-1)[0].cpu().numpy() # (B,max_length)
        roleid_preds = preds[1].argmax(-1).cpu().numpy() # (B,max_length) Dim 27
        frame_preds = preds[2].argmax(-1).cpu().numpy() # (B,max_length) Dim 752
        offset_frames_preds = frame_preds + init_frames

        B = action_preds.shape[0]
    
        # Collect events as (frame, role_id, class, score)
        raw_events = []

        for b in range(B):

            if (frame_preds[b].sum() == 0) or ((action_preds[b] == 9).sum() == 0):
                continue

            for t in range(25):

                curr_action = int(action_preds[b, t])
                curr_score  = float(score_preds[b, t])

                if (curr_action == 9) or (curr_action == 8):
                    break

                roleid = int(roleid_preds[b, t])
                frame  = int(offset_frames_preds[b, t])

                if (roleid != 26) and (0 <= frame <= end_frame):
                    cls_id = curr_action + 1   # map to [1..] (0 is background)
                    raw_events.append([frame, roleid, cls_id, curr_score])

        # Build per frame shirt lookup to handle substitutions or role changes
        T_pred = T

        # Initialize with -1 (for unknown)
        shirt_by_time = np.full((M, T_pred), -1, dtype=int)

        # Use rows that have a known shirt number, for safety...
        mask_sn = ~np.isnan(datah5[:, SHIRT_NUMBER])
        if np.any(mask_sn):
            frs_abs  = datah5[mask_sn, FRAME].astype(int)
            ltrs     = datah5[mask_sn, LEFT_TO_RIGHT].astype(int)
            rids     = datah5[mask_sn, ROLE_ID].astype(int)
            shirts   = datah5[mask_sn, SHIRT_NUMBER].astype(int)

            frs_rel = frs_abs - start_frame

            # keep only frames in [0,T_pred] and valid role ids
            valid = (frs_rel >= 0) & (frs_rel < T_pred) & (rids >= 1) & (rids <= PLAYERS_PER_TEAM)
            frs_rel, ltrs, rids, shirts = frs_rel[valid], ltrs[valid], rids[valid], shirts[valid]

            m_idxs = ltrs * PLAYERS_PER_TEAM + (rids - 1)  # 0..25
            # If multiple entries for the same (m, frame), the last seen wins
            for m_idx, fr_rel, sn in zip(m_idxs, frs_rel, shirts):
                shirt_by_time[m_idx, fr_rel] = sn

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

        # Transform to (frame, team_left_right, shirt_number, class, score) for JSON export and greedy matching for evaluation
        events = []
        for t_det_abs, m_idx, c_id, s in raw_events:
            t_rel = t_det_abs - start_frame
            if not (0 <= t_rel < T_pred):
                continue
            team_left_right = m_idx // PLAYERS_PER_TEAM
            shirt_number = int(shirt_by_time[m_idx, t_rel])
            events.append([int(t_det_abs), int(team_left_right), int(shirt_number), int(c_id), float(s)])
        
        all_events["keys"].append(k)
        all_events["events"][k] = events

        json_path = os.path.join(DATA_ROOT, r"playbyplay_PRED")
        os.makedirs(json_path, exist_ok=True)
        json_path = os.path.join(json_path, f'playbyplay_TAAD_{SET_STATUS}.json')

    out_path = os.path.join(DATA_ROOT, r"playbyplay_PRED")
    os.makedirs(out_path, exist_ok=True)
    out_path = os.path.join(out_path, f'playbyplay_TAAD_{SET_STATUS}.json')
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_events, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(all_events['keys'])} games, total events = {sum(len(v) for v in all_events['events'].values())}")
    print(f"JSON written to: {out_path}")
