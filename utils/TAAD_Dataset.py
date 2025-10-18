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
import torchvision
torchvision.disable_beta_transforms_warning()

from decord import VideoReader
from decord import cpu

import albumentations as A


class TAAD_Dataset(Dataset):

    def __init__(self,
                 data_root, 
                 set_status='train',
                 clip_length=50,
                 max_nb_samples_per_class=1000,
                 nb_tracklets=4,
                 label_dilation=1,
                 norm_m_std=(0.45, 0.225)):

        self.data_root = data_root
        self.clip_length = clip_length
        self.max_nb_samples = max_nb_samples_per_class
        self.nb_tracklets = nb_tracklets
        self.label_dilation = label_dilation
        self.set_status = set_status
        self.norm_m_std = norm_m_std

        self.sampled_frames = []
        self.sampled_plyrid = []
        self.sampled_games = []
        self.sampled_end_frames = []

        self.h5_file = None

        self.RNG = np.random.default_rng(345) # Numpy random number generator with specific seed for val and test sets

        self.train_transform = A.Compose([A.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, scale=(1.0, 1.0), rotate=(-1.5, 1.5), p=1.0),
                                          A.RandomScale(scale_limit=(-0.1,0.1)),
                                          A.PadIfNeeded(min_height=352, min_width=640),
                                          A.RandomCrop(height=352, width=640, pad_if_needed=True),
                                          A.HorizontalFlip(p=0.5),
                                          A.ColorJitter(brightness=(0.8,1.2), contrast=(0.8,1.2), saturation=(0.8,1.2), hue=(-0.1,0.1))],
                                          bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels', 'frame_indices']))


        if self.set_status == 'train' :

            self.h5_path = os.path.join(self.data_root, "data", "train_tactical_data.h5")

            with open(os.path.join(self.data_root, "data", "TAAD_sample_list.json"), "r") as f:
                self.dataset_sample_list = json.load(f)['train']
            
            self._random_sampling()

        else :

            self.h5_path = os.path.join(self.data_root, "data", "val_tactical_data.h5")

            with open(os.path.join(self.data_root, "data", "TAAD_sample_list.json"), "r") as f:
                self.dataset_sample_list = json.load(f)['val']

            self._random_sampling(random=False)


    def _random_sampling(self, random=True):

        smp_frames = []
        smp_plyrid = []
        smp_games = []
        smp_end_frames = []

        for c in range(8):

            mask = np.array(self.dataset_sample_list['events'])==c
            c_frames = np.array(self.dataset_sample_list['frames'])[mask]
            c_plyrid = np.array(self.dataset_sample_list['player_id'])[mask]
            c_games = np.array(self.dataset_sample_list['games'])[mask]
            c_endframes = np.array(self.dataset_sample_list['end_frames'])[mask]
            nb_c = c_frames.shape[0]

            if random :
                idxs = np.random.choice([i for i in range(nb_c)], min(nb_c, self.max_nb_samples), replace=False)
            else :
                idxs = self.RNG.choice([i for i in range(nb_c)], min(nb_c, self.max_nb_samples), replace=False)

            smp_frames.append(c_frames[idxs])
            smp_plyrid.append(c_plyrid[idxs])
            smp_games.append(c_games[idxs])
            smp_end_frames.append(c_endframes[idxs])

        self.sampled_frames = np.concatenate(smp_frames)
        self.sampled_plyrid = np.concatenate(smp_plyrid)
        self.sampled_games = np.concatenate(smp_games)
        self.sampled_end_frames = np.concatenate(smp_end_frames)

        if random :
            idxs = [i for i in range(len(self.sampled_frames))]
            np.random.shuffle(idxs)
            self.sampled_frames = self.sampled_frames[idxs]
            self.sampled_plyrid = self.sampled_plyrid[idxs]
            self.sampled_games = self.sampled_games[idxs]
            self.sampled_end_frames = self.sampled_end_frames[idxs]
        else :
            idxs = [i for i in range(len(self.sampled_frames))]
            self.RNG.shuffle(idxs)
            self.sampled_frames = self.sampled_frames[idxs]
            self.sampled_plyrid = self.sampled_plyrid[idxs]
            self.sampled_games = self.sampled_games[idxs]
            self.sampled_end_frames = self.sampled_end_frames[idxs]


    def resample_dataset(self, random=True):

        self._random_sampling(random)


    def __len__(self):

        return len(self.sampled_frames)
    

    def _get_clip(self, vidfilename, kept_frame_range) :

        #### FETCH IMAGE SEQUENCE ####

        vr = VideoReader(vidfilename, ctx=cpu(0))
        frames = vr.get_batch(np.asarray(kept_frame_range, dtype=np.int64)).asnumpy()
        h, w, _ = frames[0].shape

        if w != 640 or h != 352:
            resized = [cv2.resize(fr, (640, 352), interpolation=cv2.INTER_AREA) for fr in frames]
            clip = np.stack(resized, axis=0)
        else:
            clip = frames

        if clip.size == 0:
            raise RuntimeError(f"_get_clip failed for {vidfilename}")

        return clip


    def _dilate_labels(self, all_cls):

        #### LABEL DILATION ####

        M, T = all_cls.shape
        all_sharp_label = all_cls.copy()
        all_dilated_label = all_cls.copy()

        if self.label_dilation <= 0:
            return all_sharp_label, all_dilated_label

        for m in range(M):
            events = np.where(all_sharp_label[m] != 0)[0]
            for idx in events:
                cls = all_sharp_label[m, idx]
                start = max(0, idx - self.label_dilation)
                end   = min(T, idx + self.label_dilation + 1)
                all_dilated_label[m, start:end] = cls

        return all_sharp_label, all_dilated_label
        

    def __getitem__(self, index):


        FRAME, PLAYER_ID, LEFT_TO_RIGHT, SHIRT_NUMBER, ROLE_ID, X_POS, Y_POS, X_SPEED, Y_SPEED, ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT, CLS = range(14) # Indices of corresponding data from the HDF5 file


        #### CURRENT SAMPLE INFOS ####

        curr_key = self.sampled_games[index]
        curr_event_frame = self.sampled_frames[index]
        curr_player_id = self.sampled_plyrid[index]
        curr_max_frame = self.sampled_end_frames[index]
        gm_idx = curr_key.split('_')[1]
        curr_vidpath =  os.path.join(self.data_root, "videos", f'game_{gm_idx}.mp4')


        #### BUILD THE FRAME RANGE TO RETRIEVE ####
        
        possible_min_start_frame = max(0, (curr_event_frame - self.clip_length//2 - 10))
        possible_max_start_frame = min(curr_max_frame, (curr_event_frame - (self.clip_length//2 - 10)))
        possible_start_range = [i for i in range(possible_min_start_frame, possible_max_start_frame)]

        if self.set_status == 'train' :

            sf = int(np.random.choice(possible_start_range))
            ef = int(sf + self.clip_length)          

        else :

            sf = int(self.RNG.choice(possible_start_range))
            ef = int(sf + self.clip_length)
        
        kept_frame_range = [i for i in range(sf, ef)] # The list of absolute frame numbers to retrieve in the video

        assert len(kept_frame_range)==self.clip_length, f'Problem with kept_frame_range for {curr_key} - event frame : {curr_event_frame}- len = {len(kept_frame_range)}'


        #### FETCH IMAGES ####

        videoclip = self._get_clip(curr_vidpath, kept_frame_range) # Images as uint8


        #### FETCH DATA ####

        self.h5_file = h5py.File(self.h5_path, 'r')
        data = self.h5_file[curr_key][:].astype(np.float64)
        data = data[(data[:,FRAME]>=kept_frame_range[0])&(data[:,FRAME]<=kept_frame_range[-1])].copy() # Keep only the relevant slice of the tracklets


        #### SELECT NB_TRACKLETS PLAYER_IDs FOR THE SAMPLE ####

        random_player_id_list = np.unique(data[(data[:,PLAYER_ID]!=curr_player_id)&(~np.isnan(data[:,ROI_X]))][:,PLAYER_ID])
        max_nb_plyr = min(len(random_player_id_list), self.nb_tracklets)
        kept_player_ids = []

        if max_nb_plyr > 0 :
            if self.set_status == 'train' :
                kept_player_ids = list(np.random.choice(random_player_id_list, max_nb_plyr, replace=False))
            else :
                kept_player_ids = list(self.RNG.choice(random_player_id_list, max_nb_plyr, replace=False))
        
        kept_player_ids.append(curr_player_id)
        kept_player_ids.extend([300] * (self.nb_tracklets + 1 - len(kept_player_ids)))

        assert len(kept_player_ids) == (self.nb_tracklets + 1), f'Problem list of player_id for {curr_key} - event frame : {curr_event_frame} - curr_plyr {curr_player_id} - kept_players_ids: {kept_player_ids}'


        #### BUILD ROIS, MASKS, LABELS ####

        all_rois = []
        all_masks = []
        all_cls = []

        for plyr in kept_player_ids :

            tracklet_roi = []
            tracklet_mask = []
            tracklet_class = []

            if self.set_status == 'train':
                alpha = 0.125*(2.0*np.random.rand() - 1.0)
                coeff = (1.125+alpha)
            else :
                coeff = 1.125

            localdata = data[data[:,PLAYER_ID]==plyr].copy()

            for tidx, t in enumerate(kept_frame_range) :

                W, H = 640, 352
                MIN_W, MIN_H = 2, 2

                bbox = localdata[(localdata[:,FRAME]==t)&(~np.isnan(localdata[:,ROI_X]))] if len(localdata) > 0 else []

                if len(bbox) > 0 :

                    tlx = int(max(min(1920,int(bbox[0,ROI_X]-((coeff - 1.0)*bbox[0,ROI_WIDTH]//2))),0) / 3) # Adjust roi from fullHD to (352*640)
                    tly = int(max(min(1080,int(bbox[0,ROI_Y]-((coeff - 1.0)*bbox[0,ROI_HEIGHT]//2))),0) / 3.068181)
                    brx = int(max(min(1920,int(bbox[0,ROI_X]+(coeff*bbox[0,ROI_WIDTH]))),0) / 3)
                    bry = int(max(min(1080,int(bbox[0,ROI_Y]+(coeff*bbox[0,ROI_HEIGHT]))),0) / 3.068181)

                    tlx = max(0, min(W - 1, tlx))
                    tly = max(0, min(H - 1, tly))
                    brx = max(0, min(W,     brx))
                    bry = max(0, min(H,     bry))

                    if (brx - tlx) < MIN_W or (bry - tly) < MIN_H:
                        curr_roi = np.array([tidx, 100, 100, 200, 200]) # Add a dummy bbox if bbox is not fit
                        tracklet_mask.append(0.0)
                        tracklet_class.append(0)
                    else :
                        curr_roi = np.array([tidx, tlx, tly, brx, bry]) 
                        tracklet_mask.append(1.0)
                        tracklet_class.append(bbox[0,CLS])
                    
                    tracklet_roi.append(curr_roi)   
                    
                else :

                    tracklet_roi.append(np.array([tidx, 100, 100, 200, 200])) # Dummy bbox - Mask will zero-out features extracted from it.
                    tracklet_mask.append(0.0)
                    tracklet_class.append(0)

            tracklet_roi = np.stack(tracklet_roi, axis=0)
            tracklet_mask = np.array(tracklet_mask)
            tracklet_class = np.array(tracklet_class)

            assert len(tracklet_roi) > 0, f'No ROI found for {curr_key} - event frame : {curr_event_frame}'

            all_rois.append(tracklet_roi)
            all_masks.append(tracklet_mask)
            all_cls.append(tracklet_class)

        all_rois = np.stack(all_rois, axis=0) # (M,T,5)
        all_masks = np.stack(all_masks, axis=0) # (M,T)
        all_cls = np.stack(all_cls, axis=0) # (M,T)

        #### DATA AUGMENTATION : VIDEO AND BBOX ####

        if self.set_status == 'train':

            M, T, _ = all_rois.shape 
            all_bbx = all_rois[:,:,1:].reshape(-1,4) # (M*T,4) for Albumentations
            all_bbx_idx = np.arange(0,M*T,step=1) # (M*T) each bbox is traced via an integer, because Albumentations may delete some bbx
            all_frame_idx = all_rois[:,:,0].reshape(-1) # (M*T) Albumentation requirement

            augmented = self.train_transform(images=videoclip,
                                             bboxes=all_bbx,
                                             labels=all_bbx_idx,
                                             frame_indices=all_frame_idx)

            videoclip = augmented['images']
            aug_bboxes = augmented['bboxes'].astype(np.uint32)
            aug_bbx_idx = augmented['labels'].astype(np.uint32)

            newbbx = all_rois.reshape(-1,5).copy()
            newbbx[aug_bbx_idx,1:] = aug_bboxes
            all_rois = newbbx.reshape((M,T,5)) # back to (M,T,5)

            newmasks = all_masks.reshape(-1).copy()
            albmask = np.ones_like(newmasks)
            albmask[aug_bbx_idx] = 0.0
            newmasks[albmask.astype(bool)] = 0.0
            all_masks = newmasks.reshape((M,T))

            newcls = all_cls.reshape(-1).copy()
            newcls[albmask.astype(bool)] = 0
            all_cls = newcls.reshape((M,T))

        videoclip = videoclip.astype(np.float32) / 255.
        videoclip = (videoclip - self.norm_m_std[0]) / self.norm_m_std[1]
        assert len(videoclip) == self.clip_length, f"Expected L={self.clip_length}, got T={len(videoclip)}"

        #### LABEL DILATION ####

        all_sharp_cls, all_dilated_cls = self._dilate_labels(all_cls)

        #### TO TENSOR AND EXPORT ####

        all_rois = torch.from_numpy(all_rois).float()
        all_masks = torch.from_numpy(all_masks).float()
        all_sharp_cls = torch.from_numpy(all_sharp_cls).long()
        all_dilated_cls = torch.from_numpy(all_dilated_cls).long()

        return_items = []
        return_items.append(torch.from_numpy(videoclip).permute(3,0,1,2)) # (T,H,W,3) to (3,T,H,W) for input in X3D
        return_items.append(all_rois)
        return_items.append(all_masks)
        return_items.append(all_sharp_cls)
        return_items.append(all_dilated_cls)

        return return_items
