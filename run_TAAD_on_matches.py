import argparse
import numpy as np
import os
import cv2
import h5py
from tqdm import tqdm
import torch

from models.model_TAAD_baseline import X3D_TAAD_Baseline
from utils.metric_utils import post_processing_TAAD_preds_and_export

from decord import VideoReader
from decord import cpu, gpu


def parse_args():

    p = argparse.ArgumentParser(description="run TAAD Baseline and store predictions")
    p.add_argument("--model_checkpoint", type=str, default=r"D:\AI\FOOTPASS\runs\TAAD_Baseline_25092025\checkpoints\curr_model_19.pt", help="Path and filename of the checkpoint")
    p.add_argument("--data_root", type=str, default=r"D:\AI\FOOTPASS", help="Root folder with data/")
    p.add_argument("--save_path", type=str, default=r"D:\AI\FOOTPASS\TAAD_predictions", help="Folder to write TAAD predictions")
    p.add_argument("--set_to_run", type=str, default=r"train", help="Part of the dataset to run : train, val, test")
    p.add_argument("--start_index", type=int, default=12, help="Index of the first hafltime to run in the list of hdf5 file")
    p.add_argument("--end_index", type=int, default=48, help="Index of the last hafltime to run in the list of hdf5 file : -1 if running on all the set")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--clip_length", type=int, default=50)
    p.add_argument("--post_processing", type=bool, default=False)
    p.add_argument("--nms_window", type=int, default=25)

    return p.parse_args()


if __name__ == '__main__':

    args = parse_args()

    MODEL_CHECKPOINT = args.model_checkpoint
    DATA_ROOT = args.data_root
    SAVE_PATH = args.save_path
    SET_STATUS = args.set_to_run
    START_IDX = args.start_index
    END_IDX = args.end_index
    CLIP_LENGTH = args.clip_length
    BATCH_SIZE = args.batch_size
    POST_PROC = args.post_processing
    NMS_WINDOW = args.nms_window

    os.makedirs(SAVE_PATH, exist_ok=True)

    args_file = os.path.join(SAVE_PATH, "run_config.txt")
    with open(args_file, "w") as f:
        f.write("Inference configuration:\n")
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
    print(f"[INFO] Saved Inference configuration to {args_file}")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(device)


    def get_clip(vidfilename, kept_frame_range) :

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
        
        clip = clip.astype(np.float32)  # (L, 352, 640, 3), float32
        clip = clip / 255.0
        clip = (clip - 0.45) / 0.225

        return clip


    def get_roi_masks(data, local_range):

        FRAME, PLAYER_ID, LEFT_TO_RIGHT, SHIRT_NUMBER, ROLE_ID, X_POS, Y_POS, X_SPEED, Y_SPEED, ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT, CLS = range(14) # Indices of corresponding data from the HDF5 file

        roles_list = [i for i in range(1,14)] # Player roles between 1 and 13
        all_rois = []
        all_masks = []

        coeff = 1.125

        for left_to_right in [0,1] :

            for rolidx, role_id in enumerate(roles_list)  :

                tracklet_roi = []
                tracklet_mask = []
                    
                localdata = data[(data[:,LEFT_TO_RIGHT]==left_to_right)&(data[:,ROLE_ID]==role_id)].copy()

                for tidx, t in enumerate(local_range) :
                    
                    bbox = localdata[(localdata[:,FRAME]==t)&(~np.isnan(localdata[:,ROI_X]))] if len(localdata) > 0 else []

                    if len(bbox) > 0 :

                        tlx = max(min(1920,int(bbox[0,ROI_X]-((coeff - 1.0)*bbox[0,ROI_WIDTH]//2))),0)
                        tly = max(min(1080,int(bbox[0,ROI_Y]-((coeff - 1.0)*bbox[0,ROI_HEIGHT]//2))),0)
                        brx = max(min(1920,int(bbox[0,ROI_X]+(coeff*bbox[0,ROI_WIDTH]))),0)
                        bry = max(min(1080,int(bbox[0,ROI_Y]+(coeff*bbox[0,ROI_HEIGHT]))),0)

                        curr_roi = np.array([tidx, int(tlx/3), int(tly/3.068181), int(brx/3), int(bry/3.068181)]) # Adjust roi from fullHD to (352*640)

                        tracklet_roi.append(curr_roi)   
                        tracklet_mask.append(1.0)

                    else :

                        tracklet_roi.append(np.array([tidx, 100*0.5, 100*0.5, 145*0.5, 198*0.5]))
                        tracklet_mask.append(0.0)

                tracklet_roi = np.stack(tracklet_roi, axis=0)
                tracklet_mask = np.array(tracklet_mask)
                all_rois.append(tracklet_roi)
                all_masks.append(tracklet_mask)

        all_rois = np.stack(all_rois, axis=0)# (26,T,5)
        all_masks = np.stack(all_masks, axis=0) # (26,T)

        return all_rois, all_masks


    def extract_logits(game_idx_H, video_path, data_path, save_path, batch_size=32, clip_length=50) :

        B = batch_size
        M = 26 # Number of possible roles (13 roles per team)
        T = clip_length

        FRAME, PLAYER_ID, LEFT_TO_RIGHT, SHIRT_NUMBER, ROLE_ID, X_POS, Y_POS, X_SPEED, Y_SPEED, ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT, CLS = range(14) # Indices of corresponding data from the HDF5 file

        gm_idx = game_idx_H.split('_')[1]
        vidfile = os.path.join(video_path, f'game_{gm_idx}.mp4')

        h5_file = h5py.File(data_path, 'r')
        data = h5_file[game_idx_H][:].astype(np.float64)
        data = data[~np.isnan(data[:,ROI_X].astype(np.float64))].copy() # Keep rows if player is observable

        total_frame_range = np.sort(np.unique(data[:,FRAME]))
        minf = int(total_frame_range.min())
        maxf = int(total_frame_range.max())

        batch_frames = list(np.arange(minf, maxf+1, T*B))

        if batch_frames[-1] < maxf:
            batch_frames.append(maxf)

        start_frames = batch_frames[:-1]
        end_frames = batch_frames[1:]

        all_logits = []

        for idx, (sf, ef) in enumerate(zip(start_frames, end_frames)):
            
            local_range = [i for i in range(sf, ef)]

            clipimg = get_clip(vidfile, local_range)
            clipimg = torch.from_numpy(clipimg).float() # (B*T,352,640,3)
            L,H,W,_ = clipimg.shape

            if L < B*T :
                padclip = torch.zeros((B*T,H,W,3))
                padclip[:L,:,:,:] = clipimg
                clipimg = padclip
                local_range = [i for i in range(sf, sf + B*T)]

            clipimg = clipimg.cuda().half() # (B*T,352,640,3)
            clipimg = clipimg.reshape(B,T,H,W,3).permute(0,4,1,2,3) # (B,3,T,352,640)

            rois, masks = get_roi_masks(data, local_range) # (M,B*T,5), (M,B*T)
            rois = rois.reshape(M,B,T,5)
            rois[:,:,:,0] = rois[:,:,:,0] - rois[:,:,:1,0]
            rois = torch.from_numpy(rois).float().cuda().permute(1,0,2,3) # (B,M,T,5)
            masks = torch.from_numpy(masks.reshape(M,B,50)).float().cuda().permute(1,0,2) # (B,M,T)

            with torch.no_grad():

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    pred = model([clipimg, rois, masks]) # (B,9,M,T)

                preds = pred.permute(1,2,0,3).reshape(9,M,-1).half().detach().cpu().numpy() # (9,M,B*T)
                all_logits.append(preds)

        all_logits = np.concatenate(all_logits, axis=-1)
        _,_,L = all_logits.shape

        avg_logits = np.zeros((9,M,minf+L), dtype=np.float16)
        avg_logits[:,:,minf:(minf+L)] = all_logits

        #### SHIFTED ANALYSIS ####

        minf += (T//2)

        batch_frames = list(np.arange(minf, maxf+1, T*B))

        if batch_frames[-1] < maxf:
            batch_frames.append(maxf)

        start_frames = batch_frames[:-1]
        end_frames = batch_frames[1:]

        all_logits = []

        for idx, (sf, ef) in enumerate(zip(start_frames, end_frames)):
            
            local_range = [i for i in range(sf, ef)]

            clipimg = get_clip(vidfile, local_range)
            clipimg = torch.from_numpy(clipimg).float() # (B*T,352,640,3)
            L,H,W,_ = clipimg.shape

            if L < B*T :
                padclip = torch.zeros((B*T,H,W,3))
                padclip[:L,:,:,:] = clipimg
                clipimg = padclip
                local_range = [i for i in range(sf, sf + B*T)]

            clipimg = clipimg.cuda().half() # (B,3,T,352,640)
            clipimg = clipimg.reshape(B,T,H,W,3).permute(0,4,1,2,3) # (B,3,T,352,640)

            rois, masks = get_roi_masks(data, local_range) # (M,B*T,5), (M,B*T)
            rois = rois.reshape(M,B,T,5)
            rois[:,:,:,0] = rois[:,:,:,0] - rois[:,:,:1,0]
            rois = torch.from_numpy(rois).float().cuda().permute(1,0,2,3) # (B,M,T,5)
            masks = torch.from_numpy(masks.reshape(M,B,50)).float().cuda().permute(1,0,2) # (B,M,T)

            with torch.no_grad():

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    pred = model([clipimg, rois, masks]) # (B,9,M,T)

                preds = pred.permute(1,2,0,3).reshape(9,M,-1).half().detach().cpu().numpy() # (9,M,B*T)
                all_logits.append(preds)

        all_logits = np.concatenate(all_logits, axis=-1)
        _,_,L = all_logits.shape

        avg_logits[:,:,minf:(minf+L-25)] = (all_logits[:,:,:-25] + avg_logits[:,:,minf:(minf+L-25)]) / 2

        with open(os.path.join(save_path, f'avg_logits_{game_idx_H}.npy'), 'wb') as f:
            np.save(f, avg_logits)

        return 1


    ##### INSTANTIATE MODEL #####

    model = X3D_TAAD_Baseline()
    
    ckpt = torch.load(MODEL_CHECKPOINT, map_location=torch.device('cpu'))
    state_dict = ckpt['model_state_dict'] 
    model.load_state_dict(state_dict, strict=True)

    model = model.to(device)
    model.eval()


    #### MAKE PREDICTIONS ON FULL HALFTIMES #####

    h5file_path = os.path.join(DATA_ROOT, "data", f'{SET_STATUS}_tactical_data.h5')
    with h5py.File(h5file_path, 'r') as f:
        list_halftimes = list(f.keys())[START_IDX:END_IDX]

    videopath = os.path.join(DATA_ROOT, "videos")
    
    processed = []

    for gmidxH in tqdm(list_halftimes) :
        rez = extract_logits(gmidxH, videopath, h5file_path, SAVE_PATH, batch_size=BATCH_SIZE, clip_length=CLIP_LENGTH)
        processed.append((gmidxH, rez))
    
    print(processed)


    ##### POST PROCESS AND EXPORT AS JSON #####
    if POST_PROC :
        json_path = os.path.join(DATA_ROOT, r"D:\AI\FOOTPASS\playbyplay_PRED")
        os.makedirs(json_path, exist_ok=True)
        post_processing_TAAD_preds_and_export(nms_window=NMS_WINDOW, save_path=json_path, h5file_path=h5file_path, set_status=SET_STATUS)
