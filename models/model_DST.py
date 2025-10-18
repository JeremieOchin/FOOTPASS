import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def sinusoidal_positional_encoding(frame_numbers, embedding_dim):

    """
    Compute sinusoidal positional encoding for a given frame_numbers tensor.
    """

    B, T = frame_numbers.shape
    pos = frame_numbers.unsqueeze(-1)  # (B, T, 1)

    div_term = torch.exp(-torch.arange(0, embedding_dim, 2, dtype=torch.float32) * (math.log(1000.0) / embedding_dim)).cuda().float()

    pos_enc = torch.zeros(B, T, embedding_dim).cuda().float()
    pos_enc[..., 0::2] = torch.sin(pos * div_term)
    pos_enc[..., 1::2] = torch.cos(pos * div_term)
    
    return pos_enc


class DST_Logits2Events(nn.Module):

    def __init__(self, framespan, input_dim, output_dim, hidden_dim, n_heads, n_enc_layers, n_dec_layers, dropout):

        super(DST_Logits2Events, self).__init__()

        self.hidden_dim = hidden_dim
        self.framespan = framespan
        self.encoder_embedding = nn.Linear(input_dim+framespan+2, hidden_dim)
        self.decoder_embedding = nn.Linear(output_dim+framespan+2, hidden_dim)
        self.transformer = nn.Transformer(d_model=hidden_dim,
                                          nhead=n_heads,
                                          num_encoder_layers=n_enc_layers,
                                          num_decoder_layers=n_dec_layers,
                                          dropout=dropout,
                                          batch_first=True)
        
        self.action_and_roleid_output_projection = nn.Linear(hidden_dim, output_dim)
        self.timestamp_output_projection = nn.Linear(hidden_dim, framespan+2)


    def forward(self, src, tgt, src_frames, tgt_frames, src_key_padding_mask, tgt_key_padding_mask):

        B,T_tgt,_ = tgt.shape

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T_tgt,tgt.device)

        src_emb = self.encoder_embedding(src) + sinusoidal_positional_encoding(src_frames, self.hidden_dim)
        tgt_emb = self.decoder_embedding(tgt) + sinusoidal_positional_encoding(tgt_frames, self.hidden_dim)

        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        output = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        output = torch.cat([self.action_and_roleid_output_projection(output), self.timestamp_output_projection(output)], dim=-1)

        return output.permute(0, 2, 1)  # Return (B,(10+27+FRAMSPAN+2),T)


    def forward_autoregressive(self, src, src_frames, src_key_padding_mask, max_tgt_len):

        """
        Autoregressive inference function.
        
        Args:
            src (Tensor): Source sequence tensor (B, T_src, input_dim)
            src_frames (Tensor): Source frame information (B, T_src, framespan+2)
            src_key_padding_mask (Tensor): Padding mask for the source sequence (B, T_src)
            max_tgt_len (int): Maximum length of the output sequence to generate

        Returns:
            Tensor: Autoregressively generated output (B, max_tgt_len, output_dim + framespan + 2)
        """

        B, T_src, _ = src.shape
        device = src.device

        src_emb = self.encoder_embedding(src) + sinusoidal_positional_encoding(src_frames, self.hidden_dim)
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        
        # Initialize decoder input
        generated_out_action = []
        generated_out_role = []
        generated_out_frame = []

        SOS_tgt = torch.zeros((B, 1, self.action_and_roleid_output_projection.out_features), device=device)
        SOS_tgt[:,:,8] = 1
        SOS_tgt[:,:,-1] = 1
        SOS_tgt_frame = torch.zeros((B, 1, self.framespan + 2), device=device)
        SOS_tgt_frame[:,:,0] = 1

        SOS_tgt = torch.cat([SOS_tgt, SOS_tgt_frame], dim=-1)

        tgt_frames = torch.zeros((B, 1), device=device)
        
        for t in range(max_tgt_len):

            tgt_emb = self.decoder_embedding(SOS_tgt) + sinusoidal_positional_encoding(tgt_frames, self.hidden_dim)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(t + 1, device)
            
            output = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)

            out_action_and_role = self.action_and_roleid_output_projection(output[:, -1:, :])
            out_action = out_action_and_role[:,:,:10].softmax(-1)
            out_role = out_action_and_role[:,:,10:].softmax(-1)
            out_timestamp = (self.timestamp_output_projection(output[:, -1:, :])).softmax(-1)

            generated_out_action.append(out_action)
            generated_out_role.append(out_role)
            generated_out_frame.append(out_timestamp)
            
            # Update input for next step
            add_action = torch.argmax(out_action, dim=-1)
            add_role = torch.argmax(out_role, dim=-1)
            add_frame = torch.argmax(out_timestamp, dim=-1)
            add_tgt = torch.cat([torch.nn.functional.one_hot(add_action, num_classes=out_action.shape[-1]),
                                 torch.nn.functional.one_hot(add_role, num_classes=out_role.shape[-1]),
                                 torch.nn.functional.one_hot(add_frame, num_classes=out_timestamp.shape[-1])], dim=-1)
            
            SOS_tgt = torch.cat([SOS_tgt, add_tgt], dim=1)
            tgt_frames = torch.cat([tgt_frames, out_timestamp.argmax(-1)], dim=1)
        
        return torch.cat(generated_out_action, dim=1), torch.cat(generated_out_role, dim=1), torch.cat(generated_out_frame, dim=1)