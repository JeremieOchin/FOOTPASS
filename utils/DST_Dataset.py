import os
import cv2
import numpy as np
import json
import h5py
import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torchvision
torchvision.disable_beta_transforms_warning()


class DST_Dataset(Dataset):

    def __init__(self,
                 data_root, 
                 set_status='train', 
                 sequence_length=750,
                 repeat=200):

        self.data_root = data_root
        self.max_action_length = sequence_length
        self.data_path = data_root
        self.set_status = set_status
        self.repeat = repeat

        if self.set_status == 'train' :

            self.logits_h5_path = os.path.join(self.data_root, "TAAD_h5", "preds_train.h5")
            self.data_h5_path = os.path.join(self.data_root, "data", "train_tactical_data.h5")
            tmp_h5_file = h5py.File(self.data_h5_path, 'r')
            self.list_gmids = list(tmp_h5_file.keys())

            self.picklist = []
            for i in range(self.repeat):
                for gm in self.list_gmids:
                    self.picklist.append(gm)
            

        else :

            self.logits_h5_path = os.path.join(self.data_root, "TAAD_h5", "preds_val.h5")
            self.data_h5_path = os.path.join(self.data_root, "data", "val_tactical_data.h5")
            tmp_h5_file = h5py.File(self.data_h5_path, 'r')
            self.list_gmids = list(tmp_h5_file.keys())

            self.picklist = []
            for i in range(self.repeat):
                for gm in self.list_gmids:
                    self.picklist.append(gm)


        self.RNG = np.random.default_rng(345) # Numpy random number generator with specific seed for val and test sets


    def __len__(self):

        return len(self.picklist)
    

    def __getitem__(self, index):

        p_flip = np.random.random() # Used for data augmentation

        FRAME, PLAYER_ID, LEFT_TO_RIGHT, SHIRT_NUMBER, ROLE_ID, X_POS, Y_POS, X_SPEED, Y_SPEED, ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT, CLS = range(14) # Indices of corresponding data from the HDF5 file

        ##############################
        #### CURRENT SAMPLE INFOS ####

        curr_key = self.picklist[index]
        gm_idx = curr_key.split('_')[1]

        ##############################
        #### LOAD DATA AND LOGITS ####

        with h5py.File(self.data_h5_path, 'r') as data_h5_file, h5py.File(self.logits_h5_path, 'r') as logits_h5_file:
            data = data_h5_file[curr_key][:].astype(np.float64)
            logits = logits_h5_file[curr_key][:].astype(np.float32)

        data_min = int(data[:, FRAME].min())
        data_max = int(data[:, FRAME].max())
        logits_T = int(logits.shape[2])

        latest_start_data   = data_max - self.max_action_length + 1
        latest_start_logits = logits_T - self.max_action_length
        latest_start = min(latest_start_data, latest_start_logits)

        assert latest_start > data_min, f'Problem with the data from {curr_key} and predictions'

        start_frame = np.random.choice(np.arange(data_min, latest_start - 1))
        frame_range = np.arange(start_frame, (start_frame + self.max_action_length), dtype=np.int32)
        assert len(frame_range)==self.max_action_length, f'Illegal kept_frame_range for : {curr_key} - len = {len(frame_range)}'

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

        ####################################
        #### SYMMETRY DATA AUGMENTATION ####

        if (p_flip > 0.25) & (p_flip <= 0.5) & (self.set_status == 'train'):

            ### Symmetry X axis ### (team 0 => 1 / Left roles ==> right roles / X = 1 - X / X_Speed = -X_Speed) ###
            sym_mapping = np.array([13,25,17,16,15,19,18,20,21,23,22,24,14,0,12,4,3,2,6,5,7,8,10,9,11,1], dtype=np.int32)

            logits = logits[:,sym_mapping,:].copy()
            teamvec = teamvec[:,sym_mapping,:].copy()

            valid_x_mask = teamvec[0] != -15.0
            teamvec[0][valid_x_mask] = 1 - teamvec[0][valid_x_mask]
            teamvec[2][valid_x_mask] = -teamvec[2][valid_x_mask]

            ### Modify action_data
            inv_map = np.empty(sym_mapping.size, dtype=np.int32)
            inv_map[sym_mapping] = np.arange(sym_mapping.size, dtype=np.int32)

            lt = action_data[:, 1].astype(np.int32)
            rid = action_data[:, 2].astype(np.int32)
            slot = lt * 13 + (rid - 1)
            assert np.all((slot >= 0) & (slot < 26)), f"Bad slot indices in {curr_key}"
            new_slot = inv_map[slot]

            new_lt = (new_slot // 13).astype(np.int32)
            new_rid = (new_slot % 13 + 1).astype(np.int32)

            action_data[:, 1] = new_lt
            action_data[:, 2] = new_rid
                    
        elif (p_flip > 0.5) & (p_flip <= 0.75) & (self.set_status == 'train'):

            ### Symmetry Y axis ### (team 0 => 0 / Left roles ==> right roles / Y = 1 - Y) ###
            sym_mapping = np.array([0,12,4,3,2,6,5,7,8,10,9,11,1,13,25,17,16,15,19,18,20,21,23,22,24,14], dtype=np.int32)

            logits = logits[:,sym_mapping,:].copy()
            teamvec = teamvec[:,sym_mapping,:].copy()

            valid_y_mask = teamvec[0] != -15.0
            teamvec[1][valid_y_mask] = 1 - teamvec[1][valid_y_mask]
            teamvec[3][valid_y_mask] = -teamvec[3][valid_y_mask]

            ### Modify action_data
            inv_map = np.empty(sym_mapping.size, dtype=np.int32)
            inv_map[sym_mapping] = np.arange(sym_mapping.size, dtype=np.int32)

            lt = action_data[:, 1].astype(np.int32)
            rid = action_data[:, 2].astype(np.int32)
            slot = lt * 13 + (rid - 1)
            assert np.all((slot >= 0) & (slot < 26)), f"Bad slot indices in {curr_key}"
            new_slot = inv_map[slot]

            new_lt = (new_slot // 13).astype(np.int32)
            new_rid = (new_slot % 13 + 1).astype(np.int32)
            
            action_data[:, 1] = new_lt
            action_data[:, 2] = new_rid

        elif (p_flip > 0.75) & (self.set_status == 'train'):

            ### Symmetry XY axis ### (team 0 => 1 / Left roles ==> Left roles / X = 1 - X / Y = 1 - Y) ###
            sym_mapping = np.array([13,14,15,16,17,18,19,20,21,22,23,24,25,0,1,2,3,4,5,6,7,8,9,10,11,12], dtype=np.int32)

            logits = logits[:,sym_mapping,:].copy()
            teamvec = teamvec[:,sym_mapping,:].copy()

            valid_xy_mask = teamvec[0] != -15.0
            teamvec[0][valid_xy_mask] = 1 - teamvec[0][valid_xy_mask]
            teamvec[1][valid_xy_mask] = 1 - teamvec[1][valid_xy_mask]
            teamvec[2][valid_xy_mask] = -teamvec[2][valid_xy_mask]
            teamvec[3][valid_xy_mask] = -teamvec[3][valid_xy_mask]

            ### Modify action_data
            inv_map = np.empty(sym_mapping.size, dtype=np.int32)
            inv_map[sym_mapping] = np.arange(sym_mapping.size, dtype=np.int32)

            lt = action_data[:, 1].astype(np.int32)
            rid = action_data[:, 2].astype(np.int32)
            slot = lt * 13 + (rid - 1)
            assert np.all((slot >= 0) & (slot < 26)), f"Bad slot indices in {curr_key}"
            new_slot = inv_map[slot]

            new_lt = (new_slot // 13).astype(np.int32)
            new_rid = (new_slot % 13 + 1).astype(np.int32)
            
            action_data[:, 1] = new_lt
            action_data[:, 2] = new_rid

        ################################################
        #### BUILD ENCODER INPUT AND DECODER TARGET ####

        #### ENCODER INPUT ####

        teamvec = np.ascontiguousarray(np.transpose(teamvec, (2, 0, 1)).reshape(T, -1)) # (T,130)
        logits = np.ascontiguousarray(np.transpose(logits, (2, 0, 1)).reshape(T, -1)) # (T,234)

        EncoderData = torch.from_numpy(np.concatenate([teamvec, logits], axis=1)) # (T, 364)
        Encoder_abs_frame_nb = frame_range.copy()
        min_Frame = Encoder_abs_frame_nb.min()
        Encoder_abs_frame_nb -= (min_Frame - 1)

        assert len(Encoder_abs_frame_nb)>0, f'Encoder_abs_frame_nb = {len(Encoder_abs_frame_nb)}'

        #### DECODER TARGET + ADD SOS / EOS TOKENS ####

        # (T', 4) 'frame','left_to_right','short_role_id','class_id'
        N = action_data.shape[0]
        assert np.all(np.isfinite(action_data)), f"NaNs/Infs in action_data for {curr_key}"

        SOS_Token = np.zeros((1,10))
        EOS_Token = np.zeros((1,10))
        NEUTRAL_Token = np.zeros((1,27))
        SOS_Token[0,-2] = 1
        EOS_Token[0,-1] = 1
        NEUTRAL_Token[0,-1] = 1

        if N > 0 :

            Decoder_abs_frame_nb = action_data[:,0].astype(np.int32).copy()
            Decoder_abs_frame_nb -= (min_Frame - 1)
            Decoder_role_id_labels = np.zeros((N, 27), dtype=np.float32)
            Decoder_action_labels  = np.zeros((N, 10), dtype=np.float32)

            tlr   = action_data[:, 1].astype(np.int64)
            shrid = action_data[:, 2].astype(np.int64) - 1
            cls   = action_data[:, 3].astype(np.int64) - 1

            assert np.all((tlr == 0) | (tlr == 1)), f"Non-binary left_to_right in {curr_key}"

            idx = tlr * 13 + shrid

            valid = (tlr >= 0) & (tlr <= 1) & (shrid >= 0) & (shrid < 13) & (cls >= 0) & (cls < 10)
            rows = np.arange(N)[valid]

            Decoder_role_id_labels[rows, idx[valid]] = 1.0
            Decoder_action_labels[rows,  cls[valid]] = 1.0

            first_frame = np.array([0])
            last_frame = np.array([Decoder_abs_frame_nb[-1] + 1])

            Decoder_abs_frame_nb = np.concatenate([first_frame, Decoder_abs_frame_nb, last_frame], axis=0)
            Decoder_action_labels = np.concatenate([SOS_Token, Decoder_action_labels, EOS_Token], axis=0)
            Decoder_role_id_labels = np.concatenate([NEUTRAL_Token, Decoder_role_id_labels, NEUTRAL_Token], axis=0)

        else :

            Decoder_abs_frame_nb = np.array([0,Encoder_abs_frame_nb[-1]+1])
            Decoder_action_labels = np.concatenate([SOS_Token, EOS_Token], axis=0)
            Decoder_role_id_labels = np.concatenate([NEUTRAL_Token, NEUTRAL_Token], axis=0)

        assert len(Decoder_abs_frame_nb)>0, f'File {curr_key}, keptsteps {frame_range}'

        #### RETURN FEATURES AND LABELS ####

        return_items = []
        return_items.append(EncoderData) # Dim = (T,364)
        return_items.append(torch.from_numpy(Encoder_abs_frame_nb)) # Dim = (T) Absolute frame number
        return_items.append(torch.from_numpy(Decoder_role_id_labels)) # Dim = (T,27)
        return_items.append(torch.from_numpy(Decoder_abs_frame_nb)) # Dim = (T) Absolute frame number
        return_items.append(torch.from_numpy(Decoder_action_labels)) # Dim = (T,10)

        return return_items


def collate_fn(batch):

    enc_data, enc_abs_frame, dec_role_id, dec_abs_frame, dec_ac_labels = zip(*batch)

    enc_length = torch.from_numpy(np.array([tsr.shape[0] for tsr in enc_data])).long()
    dec_length = torch.from_numpy(np.array([tsr.shape[0] for tsr in dec_ac_labels])).long()

    enc_data = pad_sequence(enc_data, batch_first=True, padding_value=0)
    enc_abs_frame = pad_sequence(enc_abs_frame, batch_first=True, padding_value=-1)
    dec_role_id = pad_sequence(dec_role_id, batch_first=True, padding_value=0)
    dec_abs_frame = pad_sequence(dec_abs_frame, batch_first=True, padding_value=-1)
    dec_ac_labels = pad_sequence(dec_ac_labels, batch_first=True, padding_value=0)

    return enc_data, enc_abs_frame, enc_length, dec_role_id, dec_abs_frame, dec_ac_labels, dec_length
