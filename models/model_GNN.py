import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
from torch_geometric.nn import EdgeConv


class ChannelLayerNorm1d(nn.Module):

    def __init__(self, num_channels: int):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels)

    def forward(self, x):  # (N,C,T)
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class TAAD_GNN(nn.Module):

    """ 
    Inputs : sequences of T frames, M sequences of ROIs, M sequences of masks + graph data
        [(B,3,T,352,640), (B,M,T,5), (B,M,T), Graph Data Tensor]
    """

    def __init__(self):
        super().__init__()

        # Video backbone
        self.x3d = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_s', pretrained=True)
        self.x3d_L8 = nn.Sequential(*[self.x3d.blocks[i] for i in range(3)])
        self.up_L32 = nn.Upsample(scale_factor=(1,2,2))
        self.conv_L16_32 = nn.Conv3d(in_channels=288, out_channels=192, kernel_size=(1,3,3), padding='same', bias=False)
        self.bn_L16_32 = nn.BatchNorm3d(192)
        self.up_L16 = nn.Upsample(scale_factor=(1,2,2))
        self.conv_L8_16 = nn.Conv3d(in_channels=240, out_channels=192, kernel_size=(1,3,3), padding='same', bias=False)
        self.bn_L8_16 = nn.BatchNorm3d(192)
        self.avgpool2D = nn.AdaptiveAvgPool2d((1, 1))
        self.Dp_to_MLP = nn.Dropout(0.2)
        self.MLP_to_graph_1 = nn.Linear(in_features=192, out_features=192)
        self.MLP_to_graph_2 = nn.Linear(in_features=192, out_features=64)

        # STGNN
        self.EdgeMLP1 = nn.Linear(in_features=2*69, out_features=128)
        self.EdgeMLP2 = nn.Linear(in_features=256, out_features=128)
        self.EdgeMLP3 = nn.Linear(in_features=256, out_features=128)
        self.conv1 = EdgeConv(nn=self.EdgeMLP1, aggr='max')
        self.conv2 = EdgeConv(nn=self.EdgeMLP2, aggr='max')
        self.conv3 = EdgeConv(nn=self.EdgeMLP3, aggr='max')
        self.edge_temporal1 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, padding='same', bias=False)
        self.edge_ln1 = ChannelLayerNorm1d(128)
        self.edge_temporal2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, padding='same', bias=False)
        self.edge_ln2 = ChannelLayerNorm1d(128)
        self.edge_dp = nn.Dropout(0.1)

        # Head
        self.conv_T = nn.Conv1d(in_channels=192, out_channels=256, kernel_size=5, padding='same', bias=False)
        self.head_ln = ChannelLayerNorm1d(256)
        self.head_dp = nn.Dropout(0.2)
        self.rnn = nn.GRU(input_size=256, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.rnn_dp = nn.Dropout(0.2)
        self.fc1 = nn.Linear(256, 9)


    def _temporal_conv(self, x_n, B, T, M, conv1d, norm, dropout=None):

        x = x_n.view(B,T,M,-1)
        x = x.permute(0,2,3,1).reshape(B*M,x.shape[3], T)
        x = conv1d(x)
        x = norm(x)
        x = F.gelu(x)
        x = dropout(x)
        x = x.view(B,M,-1,T).permute(0,3,1,2).reshape(B*T*M,-1)
        return x


    def forward(self, in_x):

        x, roi, mask, data = in_x
        x_nodes, edge_index = data.x, data.edge_index

        B,_,T,_,_ = x.shape

        # VIDEO BACKBONE
        z = self.x3d_L8(x) # (B,48,T,44,80)
        y = self.x3d.blocks[3](z) # (B,96,T,22,40)
        x = self.x3d.blocks[4](y) # (B,192,T,11,20)
        x = self.up_L32(x) # (B, 192, T, 22, 40)
        x = torch.concat((x,y), dim=1) # (B,288,T,22,40)
        x = self.conv_L16_32(x) # (B,192,T,22,40)
        x = F.gelu(self.bn_L16_32(x)) # (B,192,T,22,40)
        x = self.up_L16(x) # (B,192,T,44,80)
        x = torch.concat((x,z), dim=1) # (B,240,T,44,80)
        x = self.conv_L8_16(x) # (B,192,T,44,80)
        x = F.gelu(self.bn_L8_16(x)) # (B,192,T,44,80)

        _,_,_,fh,fw = x.shape

        x = x.permute(0,2,1,3,4).reshape(-1,192,fh,fw) # (B*T,192,44,80)

        _,M,_,_ = roi.shape
        roi = roi.permute(0,2,1,3).reshape(-1,5) # (B*T*M,5)
        f_num = roi[:,0]
        batch_indices = torch.arange(B).repeat_interleave(T * M).to(roi.device)
        adjusted_frame_numbers = f_num + batch_indices * T
        roi[:,0] = adjusted_frame_numbers # (B*T*M,5)

        x = roi_align(x, roi, (4,2), 0.125) # (B*T*M,192,4,2)
        x = self.avgpool2D(x).squeeze(-1).squeeze(-1).reshape(B,T,M,192).permute(0,2,3,1).reshape(B*M,192,T) # (B*M,192,T)
        x = x*(mask.reshape(B*M,T).unsqueeze(1)) # (B*M,192,T)
        x = x.reshape(B,M,192,T).permute(0,3,1,2).reshape(B*T*M,192) # (B*T*M,192)
        x = F.relu(self.MLP_to_graph_1(x)) # (B*T*M,192)
        x = F.relu(self.MLP_to_graph_2(x)) # (B*T*M,64)

        # STGNN
        x_nodes = torch.cat([x_nodes, x], dim=-1) # (B*T*M, 69 + 64)
        x_nodes = F.relu(self.conv1(x_nodes, edge_index)) # (B*T*M, 128)
        x_nodes = self._temporal_conv(x_nodes, B, T, M, self.edge_temporal1, self.edge_ln1, self.edge_dp)
        x_nodes = F.relu(self.conv2(x_nodes, edge_index)) # (B*T*M, 128)
        x_nodes = self._temporal_conv(x_nodes, B, T, M, self.edge_temporal2, self.edge_ln2, self.edge_dp)
        x_nodes = F.relu(self.conv3(x_nodes, edge_index))  # (B*T*M,128)

        # PRED HEAD
        x = torch.cat([x, x_nodes], dim=-1) # (B*T*M,192)
        x = x.reshape(B,T,M,192).permute(0,2,3,1).reshape(B*M,192,T) # (B*M,320,T)
        x = self.conv_T(x) # (B*M,512,T)
        x = self.head_ln(x) # (B*M,512,T)
        x = F.gelu(x) # (B*M,512,T)
        x = self.head_dp(x) # (B*M,512,T)

        x = x.permute(0, 2, 1) # (B*M,T,256)
        rnn_out, _ = self.rnn(x) # (B*M,T,256)
        rnn_out = self.rnn_dp(rnn_out)
        x = rnn_out + x

        x = self.fc1(x) # (B*M,T,9)

        return x.reshape(B,M,T,9).permute(0,3,1,2) # (B,9,M,T)
