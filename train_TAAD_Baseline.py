import argparse
import numpy as np
import os
import time
from datetime import datetime
import torch
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

from models.model_TAAD_baseline import X3D_TAAD_Baseline
from utils.TAAD_Dataset import TAAD_Dataset
from utils.metric_utils import training_precision_recall

def parse_args():

    p = argparse.ArgumentParser(description="Train TAAD baseline")

    default_data_root = os.getcwd()
    default_run_path = os.path.join(default_data_root, "runs", f"TAAD_Baseline_{datetime.now():%d%m%Y}")

    p.add_argument("--data_root", type=str, default=default_data_root, help="Root folder with data/")
    p.add_argument("--run_path", type=str, default=default_run_path, help="Folder to write logs/checkpoints")

    p.add_argument("--batch_size", type=int, default=6)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--accum_steps", type=int, default=8, help="Gradient accumulation steps")

    p.add_argument("--clip_length", type=int, default=50)
    p.add_argument("--max_nb_samples_per_class", type=int, default=500)
    p.add_argument("--nb_tracklets", type=int, default=4)
    p.add_argument("--label_dilation", type=int, default=1)

    p.add_argument("--lr_x3d", type=float, default=5e-5)
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--warmup_steps", type=int, default=50)

    p.add_argument("--conf_thresh", type=float, default=0.15)
    p.add_argument("--nms_window", type=int, default=15)
    p.add_argument("--delta", type=int, default=15)

    return p.parse_args()


if __name__ == '__main__':

    args = parse_args()

    CLIP_LENGTH = args.clip_length
    MAX_NB_SAMPLES_PER_CLASS = args.max_nb_samples_per_class
    NB_TRACKLETS = args.nb_tracklets
    NB_WORKERS = args.num_workers
    LABEL_DILATION = args.label_dilation
    BATCH_SIZE = args.batch_size
    N_EPOCHS = args.epochs
    ACCUM_STEPS = args.accum_steps
    LR_X3D = args.lr_x3d
    LR_HEAD = args.lr_head
    WARMUP_STEPS = args.warmup_steps
    DATA_ROOT = args.data_root
    RUN_PATH = args.run_path
    CONF_THRESHOLD = args.conf_thresh
    NMS_WIND = args.nms_window
    DELTA = args.delta

    os.makedirs(os.path.join(RUN_PATH, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(RUN_PATH, "check"), exist_ok=True)

    args_file = os.path.join(RUN_PATH, "train_config.txt")
    with open(args_file, "w") as f:
        f.write("Training configuration:\n")
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
    print(f"[INFO] Saved training configuration to {args_file}")

    def set_x3d_freezing_schedule(model, epoch):

        if epoch <= 2:
            for name, param in model.named_parameters():
                if 'x3d' in name:
                    param.requires_grad = False
        else:
            for name, param in model.named_parameters():
                if 'x3d' in name:
                    param.requires_grad = True


    def warmup_lambda(step):

        return min(1.0, (step + 1) / WARMUP_STEPS)


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(device)

    writer = SummaryWriter()

    best_model_path = os.path.join(RUN_PATH, r'checkpoints\best_model.pt')

    ##### MODEL AND OPTIMIZER #####

    model = X3D_TAAD_Baseline()
    model = model.to(device)

    for param in model.parameters():
        param.requires_grad=True

    x3d_decay, x3d_nodecay, head_decay, head_nodecay = [], [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        group = x3d_decay if 'x3d' in name else head_decay
        group_no = x3d_nodecay if 'x3d' in name else head_nodecay
        (group_no if 'bias' in name else group).append(param)

    opt = AdamW([{'params': x3d_decay,    'lr': LR_X3D, 'weight_decay': 1e-4, 'name': 'x3d_decay'},
                 {'params': x3d_nodecay,  'lr': LR_X3D, 'weight_decay': 0.0,  'name': 'x3d_nodecay'},
                 {'params': head_decay,   'lr': LR_HEAD,'weight_decay': 1e-4, 'name': 'head_decay'},
                 {'params': head_nodecay, 'lr': LR_HEAD,'weight_decay': 0.0,  'name': 'head_nodecay'}])

    class_weights = [0.05, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95]
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    scheduler = LambdaLR(opt, lr_lambda=[warmup_lambda]*len(opt.param_groups))
    exposched = ExponentialLR(opt, 0.1)
    scaler = GradScaler()
    loss_fn = CrossEntropyLoss(reduction='none')

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

        set_x3d_freezing_schedule(model, epoch)

        training_set = TAAD_Dataset(data_root=DATA_ROOT,
                                    set_status='train',
                                    clip_length=CLIP_LENGTH,
                                    max_nb_samples_per_class=MAX_NB_SAMPLES_PER_CLASS,
                                    nb_tracklets=NB_TRACKLETS,
                                    label_dilation=LABEL_DILATION,
                                    norm_m_std=(0.45, 0.225))

        training_generator = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NB_WORKERS)

        validation_set = TAAD_Dataset(data_root=DATA_ROOT,
                                    set_status='val',
                                    clip_length=CLIP_LENGTH,
                                    max_nb_samples_per_class=MAX_NB_SAMPLES_PER_CLASS,
                                    nb_tracklets=NB_TRACKLETS,
                                    label_dilation=LABEL_DILATION,
                                    norm_m_std=(0.45, 0.225))

        validation_generator = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NB_WORKERS)

        print(f'Number of training batches : {len(training_generator)}')
        print(f'Number of validation batches : {len(validation_generator)}')

        N_batch_train = len(training_set) // BATCH_SIZE
        N_batch_val = len(validation_set) // BATCH_SIZE

        total_loss = 0

        class_names = ['background', 'drive', 'pass', 'cross', 'throw-in', 'shot', 'header', 'tackle', 'block']

        # ============================
        # TRAINING LOOP
        # ============================
        

        for batch_idx, (x_batch, roi_batch, masks, sharp_labels, dilated_labels) in enumerate(training_generator):

            B,_,T,H,W = x_batch.shape
            _,M,_,_ = roi_batch.shape

            tr_labels = []
            tr_preds = []

            ### PRED, LOSS, BACKWARD, STEP ###

            x_batch = x_batch.to(device, non_blocking=True).half() # (B,3,T,544,960)
            roi_batch = roi_batch.to(device, non_blocking=True).float() # (B,M,T,5)
            masks = masks.to(device, non_blocking=True).float() # (B,M,T)
            dilated_labels = dilated_labels.to(device, non_blocking=True).long() # (B,M,T)

            expanded_weights = (class_weights[dilated_labels]).float() # (B,M,T)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                pred = model([x_batch, roi_batch, masks]) # (B,9,M,T)
                loss = (loss_fn(pred, dilated_labels)*masks*expanded_weights).mean() # (B,M,T).mean()
                total_loss += loss.item()

            scaler.scale(loss / ACCUM_STEPS).backward()

            if (batch_idx + 1) % ACCUM_STEPS == 0 or (batch_idx + 1) == len(training_generator):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                if curr_iter <= WARMUP_STEPS:
                    scheduler.step()

            writer.add_scalars('lr/all_groups',{g.get('name', f'group_{i}'): g['lr'] for i, g in enumerate(opt.param_groups)}, curr_iter)
            writer.add_scalar('CE Loss/Total', loss.item(), curr_iter)
            
            ### CALCULATE PER CLASS ACCURACY ###

            preds = pred.softmax(1).permute(1,0,2,3).detach().cpu().numpy() # (9,B,M,T)
            sharp_labels = sharp_labels.numpy() # (B,M,T)

            tr_preds.append(preds)
            tr_labels.append(sharp_labels)

            if (batch_idx + 1) % ACCUM_STEPS == 0 or (batch_idx + 1) == len(training_generator):

                tr_labels = np.concatenate(tr_labels, axis=0)
                tr_preds = np.concatenate(tr_preds, axis=1)
                precision, recall = training_precision_recall(preds=tr_preds, sharp_labels=tr_labels, conf_thresh=CONF_THRESHOLD, nms_window=NMS_WIND, delta=DELTA)

                tr_labels = []
                tr_preds = []

                for c in range(1, len(precision)):

                    writer.add_scalar(f'Precision/Train/{class_names[c]}', precision[c], curr_iter)
                    writer.add_scalar(f'Recall/Train/{class_names[c]}',    recall[c],    curr_iter)
                
                prec_str = " ".join([f"C{c}:{precision[c]:.2f}" for c in range(1, len(precision))])
                rec_str  = " ".join([f"C{c}:{recall[c]:.2f}" for c in range(1, len(recall))])

                print(f"Train Epoch: {epoch:>4} - {(batch_idx+1):>4}/{len(training_generator)}"
                    f"\tLoss: {loss.item():.7f}\n"
                    f"    Precision -> {prec_str}\n"
                    f"    Recall    -> {rec_str}")
   
            curr_iter += 1

        current_model_path = os.path.join(RUN_PATH, f'checkpoints\curr_model_{epoch}.pt')
        current_training_loss = loss
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'loss': current_training_loss,
                    }, current_model_path)

        avg_loss = total_loss / len(training_generator)
        writer.add_scalar('EpochLoss/Train', avg_loss, epoch)

        if (epoch % 10 == 0) and (epoch > 5):
            exposched.step()

        # ============================
        # VALIDATION LOOP
        # ============================

        val_labels = []
        val_preds = []

        with torch.no_grad():

            model.eval()
            val_loss = 0

            for batch_idx, (x_batch, roi_batch, masks, sharp_labels, dilated_labels) in enumerate(validation_generator):

                B,_,T,H,W = x_batch.shape
                _,M,_,_ = roi_batch.shape

                # Transfer to GPU
                x_batch = x_batch.to(device, non_blocking=True).half() # (B,3,T,544,960)
                roi_batch = roi_batch.to(device, non_blocking=True).float() # (B,M,T,5)
                masks = masks.to(device, non_blocking=True).float() # (B,M,T)
                dilated_labels = dilated_labels.to(device, non_blocking=True).long() # (B,M,T)

                expanded_weights = (class_weights[dilated_labels]).float() # (B,M,T)

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    pred = model([x_batch, roi_batch, masks]) # (B,9,M,T)
                    vloss = (loss_fn(pred, dilated_labels)*masks*expanded_weights).mean() # (B,M,T).mean()
                    val_loss += vloss.item()

                preds = pred.softmax(1).permute(1,0,2,3).detach().cpu().numpy() # (9,B,M,T)
                labs = sharp_labels.detach().cpu().numpy() # (B,M,T)

                val_preds.append(preds)
                val_labels.append(labs)

                if batch_idx % 10 == 0:
                    print(f'\nValidation Epoch: {epoch} - {(batch_idx+1):>4} / {N_batch_val}')

            print(f'\nValidation set: Summed loss over Validation dataset: {val_loss}\n')

            writer.add_scalar('EpochLoss/ValTotal', val_loss / len(validation_generator), epoch)

            if val_loss < best_validation_loss_value :

                torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': opt.state_dict(),
                            'loss': current_training_loss,
                            }, best_model_path)
                
                best_validation_loss_value = val_loss

        val_labels = np.concatenate(val_labels, axis=0)
        val_preds = np.concatenate(val_preds, axis=1)

        val_prec, val_rec = training_precision_recall(preds=val_preds, sharp_labels=val_labels, conf_thresh=0.15, nms_window=15, delta=15)

        for c in range(1, len(val_prec)):
            writer.add_scalar(f'Precision/Val/{class_names[c]}', val_prec[c], epoch)
            writer.add_scalar(f'Recall/Val/{class_names[c]}',    val_rec[c],  epoch)

        prec_str = " ".join([f"C{c}:{val_prec[c]:.2f}" for c in range(1, len(val_prec))])
        rec_str  = " ".join([f"C{c}:{val_rec[c]:.2f}"  for c in range(1, len(val_rec))])
        print(f"Validation Epoch: {epoch:>4}  Loss: {val_loss/len(validation_generator):.7f}\n"
            f"    Val Precision -> {prec_str}\n"
            f"    Val Recall    -> {rec_str}")

        with open(os.path.join(RUN_PATH, r'check', f'val_preds_epoch_{epoch}.npy'), 'wb') as f :
            np.save(f, val_preds)

        with open(os.path.join(RUN_PATH, r'check', f'val_labels_epoch_{epoch}.npy'), 'wb') as f :
            np.save(f, val_labels)

    writer.flush()
    writer.close()