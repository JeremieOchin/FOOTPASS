import argparse
import numpy as np
import os
import time
import torch
import pickle
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

from models.model_DST import DST_Logits2Events
from utils.DST_Dataset import DST_Dataset, collate_fn
from utils.metric_utils import  calculate_precision_recall, build_role_sequences, build_action_sequences

def parse_args():

    p = argparse.ArgumentParser(description="Train DST Logits2Events")

    default_data_root = os.getcwd()
    default_run_path = os.path.join(default_data_root, "runs", f"DST_Baseline")

    p.add_argument("--data_root", type=str, default=default_data_root, help="Root folder with data/")
    p.add_argument("--run_path", type=str, default=default_run_path, help="Folder to write logs/checkpoints")

    p.add_argument("--batch_size", type=int, default=96)
    p.add_argument("--num_workers", type=int, default=12)
    p.add_argument("--epochs", type=int, default=15)

    p.add_argument("--framespan", type=int, default=750)
    p.add_argument("--inputdim", type=int, default=364)
    p.add_argument("--outputdim", type=int, default=37)
    p.add_argument("--hiddendim", type=int, default=512)
    p.add_argument("--numlayers", type=int, default=6)
    p.add_argument("--numheads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=0.00025)
    p.add_argument("--warmup_steps", type=int, default=1000)

    p.add_argument("--delta", type=int, default=15)

    return p.parse_args()

if __name__ == '__main__':

    args = parse_args()

    FRAMESPAN = args.framespan
    INPUT_DIM = args.inputdim
    OUTPUT_DIM = args.outputdim
    HIDDEN_DIM = args.hiddendim
    NUM_LAYERS = args.numlayers
    NUM_HEADS = args.numheads
    DROPOUT = args.dropout
    BATCH_SIZE = args.batch_size
    NB_WORKERS = args.num_workers
    N_EPOCHS = args.epochs
    LR = args.lr
    WARMUP_STEPS = args.warmup_steps
    DATA_ROOT = args.data_root
    RUN_PATH = args.run_path
    DELTA = args.delta

    os.makedirs(os.path.join(RUN_PATH, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(RUN_PATH, "check"), exist_ok=True)

    args_file = os.path.join(RUN_PATH, "train_config.txt")
    with open(args_file, "w") as f:
        f.write("Training configuration:\n")
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
    print(f"[INFO] Saved training configuration to {args_file}")


    def warmup_lambda(step):
        return min(1.0, (step + 1) / WARMUP_STEPS)


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


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(device)

    writer = SummaryWriter()

    best_model_path = os.path.join(RUN_PATH, r'checkpoints\best_model.pt')


    ##### DATASET ######

    training_set = DST_Dataset(data_root=DATA_ROOT, set_status='train', sequence_length=FRAMESPAN, repeat=2000)
    training_generator = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NB_WORKERS, collate_fn=collate_fn)

    validation_set = DST_Dataset(data_root=DATA_ROOT, set_status='val', sequence_length=FRAMESPAN, repeat=750)
    validation_generator = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NB_WORKERS, collate_fn=collate_fn)

    print(f'[INFO] Number of training batches : {len(training_generator)}')
    print(f'[INFO] Number of validation batches : {len(validation_generator)}')


    ##### MODEL AND OPTIMIZER #####

    model = DST_Logits2Events(framespan=FRAMESPAN,
                              input_dim=INPUT_DIM,
                              output_dim=OUTPUT_DIM,
                              hidden_dim=HIDDEN_DIM,
                              n_heads=NUM_HEADS,
                              n_enc_layers=NUM_LAYERS,
                              n_dec_layers=NUM_LAYERS,
                              dropout=DROPOUT)

    model = model.to(device)

    for param in model.parameters():
        param.requires_grad=True

    no_weight_decay_prm_grp = []
    weight_decay_prm_grp = []

    for name, p in model.named_parameters():
        if 'bias' in name:
            no_weight_decay_prm_grp.append(p)
        else:
            weight_decay_prm_grp.append(p)

    opt = AdamW(no_weight_decay_prm_grp, lr=LR) 
    opt.add_param_group({'params': weight_decay_prm_grp, 'weight_decay':1e-04})

    CE_Action_loss_fn = CrossEntropyLoss(reduction='none', label_smoothing=0.05)
    CE_RoleID_loss_fn = CrossEntropyLoss(reduction='none', label_smoothing=0.05)
    CE_Frame_loss_fn = CrossEntropyLoss(reduction='none')

    exposched = ExponentialLR(opt, 0.1)
    scheduler = LambdaLR(opt, lr_lambda=warmup_lambda)

    scaler = GradScaler()

    curr_iter = 0
    val_iter = 0
    best_validation_loss_value = 1e6

    # ============================
    # GLOBAL LOOP
    # ============================

    print('Start training')

    model.train()
    opt.zero_grad() # set_to_none=True

    for epoch in range(1, N_EPOCHS + 1):

        model.train()

        N_batch_train = len(training_set) // BATCH_SIZE
        N_batch_val = len(validation_set) // BATCH_SIZE

        total_loss = 0

        class_names = ['background', 'drive', 'pass', 'cross', 'throw-in', 'shot', 'header', 'tackle', 'block']

        # ============================
        # TRAINING LOOP
        # ============================
        

        for batch_idx, (enc_data, enc_abs_frame, enc_length, dec_role_id, dec_abs_frame, dec_ac_labels, dec_length) in enumerate(training_generator):

            ENC_B, ENC_T = enc_abs_frame.shape
            DEC_B, DEC_T = dec_abs_frame.shape

            # Prepare data
            src_key_padding_mask = (enc_abs_frame == -1) 
            tgt_key_padding_mask = (dec_abs_frame == -1)

            loss_mask = (~tgt_key_padding_mask).float()[:,1:]

            src = torch.cat([enc_data, one_hot_encode_with_padding(enc_abs_frame,FRAMESPAN)], dim=-1).float() # (B,T,234+130+FRAMESPAN+2)
            tgt_action_in = torch.cat([dec_ac_labels[:, :-1, :].float(), dec_role_id[:, :-1, :].float(), one_hot_encode_with_padding(dec_abs_frame,FRAMESPAN)[:, :-1, :]], dim=-1).float()  # (B,:T-1,10+27+FRAMESPAN+2) Teacher forcing: input is everything except EOS
            tgt_actions_out = dec_ac_labels[:, 1:, :].argmax(-1).long()  # Target output skips SOS
            tgt_role_id_out = dec_role_id[:, 1:, :].argmax(-1).long()
            tgt_frame_out = (one_hot_encode_with_padding(dec_abs_frame,FRAMESPAN)[:, 1:, :]).argmax(-1).long()

            # Transfer to GPU
            src = src.to(device, non_blocking=True).float() #.half() # (B,T,234+130+FRAMESPAN+2)
            tgt_action_in = tgt_action_in.to(device, non_blocking=True).float() # (B,T,10+27+FRAMESPAN+2)
            tgt_actions_out = tgt_actions_out.to(device, non_blocking=True) # (B,T)
            tgt_role_id_out = tgt_role_id_out.to(device, non_blocking=True) # (B,T)
            tgt_frame_out = tgt_frame_out.to(device, non_blocking=True) # (B,T)
            enc_abs_frame = enc_abs_frame.to(device, non_blocking=True).float() # (B,T)
            dec_abs_frame = dec_abs_frame.to(device, non_blocking=True).float() # (B,T)
            src_key_padding_mask = src_key_padding_mask.to(device, non_blocking=True) # (B,T)
            tgt_key_padding_mask = tgt_key_padding_mask.to(device, non_blocking=True) # (B,T)
            
            loss_mask = loss_mask.to(device, non_blocking=True).float()

            pred = model(src, tgt_action_in, enc_abs_frame, dec_abs_frame[:, :-1], src_key_padding_mask, tgt_key_padding_mask[:,:-1]) # (B,10+27+FRAMSPAN+2,T)
            pred_action = pred[:,:10,:] # (B,10,T)
            pred_role_id = pred[:,10:37,:] # (B,27,T)
            pred_time = pred[:,37:,:] # (B,FRAMESPAN+2,T)
            action_loss = (CE_Action_loss_fn(pred_action, tgt_actions_out) * loss_mask ).mean()
            roleid_loss = (CE_RoleID_loss_fn(pred_role_id, tgt_role_id_out) * loss_mask ).mean()
            frame_loss = (CE_Frame_loss_fn(pred_time,tgt_frame_out) * loss_mask).mean()
            loss = action_loss + roleid_loss + frame_loss
            total_loss += loss.item()

            loss.backward()
            opt.step()
            opt.zero_grad() # set_to_none=True

            if epoch == 1 :
                scheduler.step()

            writer.add_scalars('lr/all_groups',{g.get('name', f'group_{i}'): g['lr'] for i, g in enumerate(opt.param_groups)}, curr_iter)
   
            ### CALCULATE PER CLASS ACCURACY ###

            preds = pred_action.softmax(1).permute(1,0,2).detach().cpu().numpy() # (10,B,T)
            roles = pred_role_id.softmax(1).permute(1,0,2).detach().cpu().numpy() # (27,B,T)
            predframes = pred_time.softmax(1).argmax(1).detach().cpu().numpy() # (1500,B,T) to (B,T)
            labs = tgt_actions_out.detach().cpu().numpy() # (B,T)
            labroles = tgt_role_id_out.detach().cpu().numpy() # (B,T)
            labframes = dec_abs_frame.detach().cpu().numpy()[:,1:] # (B,T)

            predicted_classes = np.argmax(preds, axis=0)
            predicted_roles = np.argmax(roles, axis=0)
            predseq, labelseq = build_action_sequences(predicted_classes, predframes, labs, labframes)
            overall_action_accuracy, overall_action_recall = calculate_precision_recall(predseq, labelseq, DELTA)
            predseq, labelseq = build_role_sequences(predicted_roles, predframes, labroles, labframes)
            overall_role_accuracy, overall_role_recall = calculate_precision_recall(predseq, labelseq, DELTA)

            writer.add_scalar('CE Action loss per iter / train', action_loss.item(), curr_iter)
            writer.add_scalar('CE RoleID loss per iter / train', roleid_loss.item(), curr_iter)
            writer.add_scalar('CE Frame loss per iter / train', frame_loss.item(), curr_iter)
            writer.add_scalar('TOTAL loss per iter / train', loss.item(), curr_iter)
            writer.add_scalar('overall_action_accuracy per iter / train', overall_action_accuracy, curr_iter)
            writer.add_scalar('overall_action_recall per iter / train', overall_action_recall, curr_iter)
            writer.add_scalar('overall_role_accuracy per iter / train', overall_role_accuracy, curr_iter)
            writer.add_scalar('overall_role_recall per iter / train', overall_role_recall, curr_iter)

            curr_iter += 1
 
            print(f'Train Epoch: {epoch:>4} - {(batch_idx+1):>4} / {N_batch_train} \tAction CE Loss : {(action_loss.item()):.7f} \tRoleID CE Loss : {(roleid_loss.item()):.7f} \tFrame CE Loss : {(frame_loss.item()):.7f}\n') 
            
        current_model_path = os.path.join(RUN_PATH, f'checkpoints\curr_model_{epoch}.pt')
        current_training_loss = loss
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'loss': current_training_loss,
                    }, current_model_path)

        writer.add_scalar('TOTAL loss per epoch / train', total_loss/len(training_generator), epoch)

        if epoch in [3,6,8]:
            exposched.step()

        # ============================
        # VALIDATION LOOP
        # ============================

        val_labels = []
        val_labroles = []
        val_labframes = []
        val_preds = []
        val_pred_roles = []
        val_predframes = []
        val_noisyac = []
        val_noisyrole = []
        val_noisyframes = []

        with torch.no_grad():

            model.eval()

            val_loss = 0
            val_action_loss = 0
            val_roleid_loss = 0
            val_frame_loss = 0

            for batch_idx, (enc_data, enc_abs_frame, enc_length, dec_role_id, dec_abs_frame, dec_ac_labels, dec_length) in enumerate(validation_generator):

                ENC_B, ENC_T = enc_abs_frame.shape
                DEC_B, DEC_T = dec_abs_frame.shape

                # Prepare data
                src_key_padding_mask = (enc_abs_frame == -1) 
                tgt_key_padding_mask = (dec_abs_frame == -1)

                loss_mask = (~tgt_key_padding_mask).float()[:,1:]

                src = torch.cat([enc_data, one_hot_encode_with_padding(enc_abs_frame,FRAMESPAN)], dim=-1).float() # (B,T,234+130+FRAMESPAN+2)
                tgt_action_in = torch.cat([dec_ac_labels[:, :-1, :].float(), dec_role_id[:, :-1, :].float(), one_hot_encode_with_padding(dec_abs_frame,FRAMESPAN)[:, :-1, :]], dim=-1).float()  # (B,:T-1,234+130+FRAMESPAN+2) Teacher forcing: input is everything except EOS
                tgt_actions_out = dec_ac_labels[:, 1:, :].argmax(-1).long()  # Target output skips SOS
                tgt_role_id_out = dec_role_id[:, 1:, :].argmax(-1).long()
                tgt_frame_out = (one_hot_encode_with_padding(dec_abs_frame,FRAMESPAN)[:, 1:, :]).argmax(-1).long()

                # Transfer to GPU
                src = src.to(device, non_blocking=True).float() #.half() # (B,T,234+130+FRAMESPAN+2)
                tgt_action_in = tgt_action_in.to(device, non_blocking=True).float() # (B,T,10+27+FRAMESPAN+2)
                tgt_actions_out = tgt_actions_out.to(device, non_blocking=True) # (B,T)
                tgt_role_id_out = tgt_role_id_out.to(device, non_blocking=True) # (B,T)
                tgt_frame_out = tgt_frame_out.to(device, non_blocking=True) # (B,T)
                enc_abs_frame = enc_abs_frame.to(device, non_blocking=True).float() # (B,T)
                dec_abs_frame = dec_abs_frame.to(device, non_blocking=True).float() # (B,T)
                src_key_padding_mask = src_key_padding_mask.to(device, non_blocking=True) # (B,T)
                tgt_key_padding_mask = tgt_key_padding_mask.to(device, non_blocking=True) # (B,T)
                
                loss_mask = loss_mask.to(device, non_blocking=True).float()

                pred = model(src, tgt_action_in, enc_abs_frame, dec_abs_frame[:, :-1], src_key_padding_mask, tgt_key_padding_mask[:,:-1]) # (B,10+27+FRAMSPAN+2,T)
                pred_action = pred[:,:10,:] # (B,10,T)
                pred_role_id = pred[:,10:37,:] # (B,27,T)
                pred_time = pred[:,37:,:] # (B,FRAMESPAN+2,T)
                action_loss = (CE_Action_loss_fn(pred_action, tgt_actions_out) * loss_mask ).mean()
                roleid_loss = (CE_RoleID_loss_fn(pred_role_id, tgt_role_id_out) * loss_mask ).mean()
                frame_loss = (CE_Frame_loss_fn(pred_time,tgt_frame_out) * loss_mask).mean()
                vloss = action_loss + roleid_loss + frame_loss
                val_loss += vloss.item()

                val_action_loss += action_loss.item()
                val_roleid_loss += roleid_loss.item()
                val_frame_loss += frame_loss.item()
                vloss = action_loss + frame_loss
                
                preds = pred_action.softmax(1).permute(1,0,2).detach().cpu().numpy() # (10,B,T)
                roles = pred_role_id.softmax(1).permute(1,0,2).detach().cpu().numpy() # (27,B,T)
                predframes = pred_time.softmax(1).argmax(1).detach().cpu().numpy() # (1500,B,T) to (B,T)
                labs = tgt_actions_out.detach().cpu().numpy() # (B,T)
                labroles = tgt_role_id_out.detach().cpu().numpy() # (B,T)
                labframes = dec_abs_frame.detach().cpu().numpy()[:,1:] # (B,T)

                predicted_classes = np.argmax(preds, axis=0)
                predicted_roles = np.argmax(roles, axis=0)
                predseq, labelseq = build_action_sequences(predicted_classes, predframes, labs, labframes)
                overall_action_accuracy, overall_action_recall = calculate_precision_recall(predseq, labelseq, DELTA)
                predseq, labelseq = build_role_sequences(predicted_roles, predframes, labroles, labframes)
                overall_role_accuracy, overall_role_recall = calculate_precision_recall(predseq, labelseq, DELTA)

                val_labels.append(labs)
                val_labroles.append(labroles)
                val_labframes.append(labframes)
                val_preds.append(preds)
                val_pred_roles.append(roles)
                val_predframes.append(predframes)

                print(f'\nValidation Epoch: {epoch} - {(batch_idx+1):>4} / {N_batch_val}')

                writer.add_scalar('overall_action_accuracy per iter / val', overall_action_accuracy, val_iter)
                writer.add_scalar('overall_action_recall per iter / val', overall_action_recall, val_iter)
                writer.add_scalar('overall_role_accuracy per iter / val', overall_role_accuracy, val_iter)
                writer.add_scalar('overall_role_recall per iter / val', overall_role_recall, val_iter)

                val_iter += 1

            print(f'\nValidation set: Summed loss over Validation dataset: {val_loss}\n')

            writer.add_scalar('CE Action loss per epoch / val', val_action_loss/len(validation_generator), epoch)
            writer.add_scalar('CE Frame loss per epoch / val', val_frame_loss/len(validation_generator), epoch)
            writer.add_scalar('TOTAL loss per epoch / val', val_loss/len(validation_generator), epoch)

            if val_loss < best_validation_loss_value :

                torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': opt.state_dict(),
                            'loss': current_training_loss,
                            }, best_model_path)
                
                print(f'{time.ctime()} - Best model saved ! \n')

                best_validation_loss_value = val_loss


        with open(os.path.join(RUN_PATH, r'check', f'val_preds_epoch_{epoch}.pkl'), 'wb') as f :
            pickle.dump(val_preds, f)

        with open(os.path.join(RUN_PATH, r'check', f'val_pred_roles_epoch_{epoch}.pkl'), 'wb') as f :
            pickle.dump(val_pred_roles, f)

        with open(os.path.join(RUN_PATH, r'check', f'val_predframes_epoch_{epoch}.pkl'), 'wb') as f :
            pickle.dump(val_predframes, f)

        with open(os.path.join(RUN_PATH, r'check', f'val_labels_epoch_{epoch}.pkl'), 'wb') as f :
            pickle.dump(val_labels, f)

        with open(os.path.join(RUN_PATH, r'check', f'val_lab_roles_epoch_{epoch}.pkl'), 'wb') as f :
            pickle.dump(val_labroles, f)

        with open(os.path.join(RUN_PATH, r'check', f'val_labframes_epoch_{epoch}.pkl'), 'wb') as f :
            pickle.dump(val_labframes,f)

    writer.flush()
    writer.close()