import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.ops import roi_align


class X3D_TAAD_Baseline(nn.Module):

    """ 
    Inputs : sequences of T frames, M sequences of ROIs, M sequences of masks
        [(B,3,T,352,640), (B,M,T,5), (B,M,T)]
    """

    def __init__(self):
        super().__init__()

        self.x3d = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_s', pretrained=True)
        self.x3d_L4 = nn.Sequential(*[self.x3d.blocks[i] for i in range(2)])
        self.up_L32 = nn.Upsample(scale_factor=(1,2,2))
        self.conv_L16_32 = nn.Conv3d(in_channels=288, out_channels=192, kernel_size=(1,3,3), padding='same', bias=False)
        self.bn_L16_32 = nn.BatchNorm3d(192)
        self.up_L16 = nn.Upsample(scale_factor=(1,2,2))
        self.conv_L8_16 = nn.Conv3d(in_channels=240, out_channels=192, kernel_size=(1,3,3), padding='same', bias=False)
        self.bn_L8_16 = nn.BatchNorm3d(192)
        self.avgpool2D = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv1d(in_channels=192, out_channels=512, kernel_size=3, padding='same', bias=False)
        self.bn1 = nn.BatchNorm1d(num_features=512)
        self.fc1 = nn.Linear(512,9)

        
    def forward(self, in_x):

        x, roi, mask = in_x
        b,c,l,h,w = x.shape

        w = self.x3d_L4(x) # (B, 48, T, 88, 160)
        z = self.x3d.blocks[2](w) # (B, 48, T, 44, 80)
        y = self.x3d.blocks[3](z) # (B, 96, T, 22, 40)
        x = self.x3d.blocks[4](y) # (B, 192, T, 11, 20)
        x = self.up_L32(x) # (B, 192, T, 22, 40)
        x = torch.concat((x,y), dim=1) # (B, 288, T, 22, 40)
        x = self.conv_L16_32(x) # (B, 192, T, 22, 40)
        x = F.gelu(self.bn_L16_32(x)) # (B, 192, T, 22, 40)
        x = self.up_L16(x) # (B, 192, T, 44, 80)
        x = torch.concat((x,z), dim=1) # (B, 240, T, 44, 80)
        x = self.conv_L8_16(x) # (B, 192, T, 44, 80)
        x = F.gelu(self.bn_L8_16(x)) # (B, 192, T, 44, 80)

        _,_,_,fh,fw = x.shape

        x = x.permute(0,2,1,3,4).reshape(-1,192,fh,fw) # (B*T,192,44,80)

        _,M,_,_ = roi.shape
        roi = roi.permute(0,2,1,3).reshape(-1,5) # (B*T*M,5)
        f_num = roi[:,0]
        batch_indices = torch.arange(b).repeat_interleave(l * M).cuda()
        adjusted_frame_numbers = f_num + batch_indices * l
        roi[:,0] = adjusted_frame_numbers # (B*T*M,5)

        x = roi_align(x, roi, (4,2), 0.125) # (B*T*M, 192, 4, 2)
        x = self.avgpool2D(x).squeeze(-1).squeeze(-1).reshape(b,l,M,192).permute(0,2,3,1).reshape(b*M,192,l) # (B*T*M, 192, 1, 1) to (B*M, 192, T)
        x = x*(mask.reshape(b*M,l).unsqueeze(1)) #  (B*M,192,T)
        x = F.gelu(self.bn1(self.conv1(x))) #  (B*M,512,T)

        x = self.fc1(x.permute(0,2,1)) # (B*M,T,9)

        return x.reshape(b,M,l,9).permute(0,3,1,2) # (B*M,T,9) to (B,9,M,T)

