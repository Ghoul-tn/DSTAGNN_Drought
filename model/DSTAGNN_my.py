# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib.utils import scaled_Laplacian, cheb_polynomial


class SScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(SScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.
        return scores


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, num_of_d):
        super().__init__()
        self.d_k = d_k
        self.num_of_d = num_of_d
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, attn_mask, res_att):
        """
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v, d_v]
        res_att: [batch_size, n_heads, len_q, len_k] or None
        """
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        # Handle residual attention
        if res_att is not None:
            # Ensure compatible shapes
            if res_att.dim() == 4:
                if res_att.size(1) != scores.size(1):  # Head dimension mismatch
                    res_att = res_att.repeat(1, scores.size(1)//res_att.size(1), 1, 1)
                scores = scores + res_att
        
        # Apply mask if provided
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)
            
        # Compute attention weights
        attn = self.softmax(scores)
        
        # Compute context
        context = torch.matmul(attn, V)
        
        return context, attn

class SMultiHeadAttention(nn.Module):
    def __init__(self, DEVICE, d_model, d_k ,d_v, n_heads):
        super(SMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.DEVICE = DEVICE
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)

    def forward(self, input_Q, input_K, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1,
                                                  1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        attn = SScaledDotProductAttention(self.d_k)(Q, K, attn_mask)
        return attn

class MultiHeadAttention(nn.Module):
    def __init__(self, DEVICE, d_model, d_k, d_v, n_heads, num_of_d):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.num_of_d = num_of_d
        self.DEVICE = DEVICE
        
        # Projection matrices
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model).to(DEVICE)

    def forward(self, input_Q, input_K, input_V, attn_mask, res_att):
        batch_size = input_Q.size(0)
        residual = input_Q
        
        # Linear projections
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        
        # Prepare attention mask
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        
        # Handle residual attention shape
        if res_att is not None and res_att.dim() == 4:
            if res_att.size(1) != self.n_heads:
                res_att = res_att.repeat(1, self.n_heads//res_att.size(1), 1, 1)
        
        # Scaled dot-product attention
        context, attn = ScaledDotProductAttention(self.d_k, self.num_of_d)(
            Q, K, V, attn_mask, res_att)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.n_heads * self.d_v)
        
        # Final linear projection
        output = self.fc(context)
        
        return self.norm(output + residual), attn


class cheb_conv_withSAt(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, K, cheb_polynomials, in_channels, out_channels, num_of_vertices):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv_withSAt, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.relu = nn.ReLU(inplace=True)
        self.Theta = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])
        self.mask = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(num_of_vertices,num_of_vertices).to(self.DEVICE)) for _ in range(K)])
    def forward(self, x, spatial_attention, adj_pa):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)

            for k in range(self.K):

                T_k = self.cheb_polynomials[k]  # (N,N)
                mask = self.mask[k]

                myspatial_attention = spatial_attention[:, k, :, :] + adj_pa.mul(mask)
                myspatial_attention = F.softmax(myspatial_attention, dim=1)

                T_k_with_at = T_k.mul(myspatial_attention)   # (N,N)*(N,N) = (N,N) 多行和为1, 按着列进行归一化

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = T_k_with_at.permute(0, 2, 1).matmul(graph_signal)  # (N, N)(b, N, F_in) = (b, N, F_in) 因为是左乘，所以多行和为1变为多列和为1，即一行之和为1，进行左乘

                output = output + rhs.matmul(theta_k)  # (b, N, F_in)(F_in, F_out) = (b, N, F_out)

            outputs.append(output.unsqueeze(-1))  # (b, N, F_out, 1)

        return self.relu(torch.cat(outputs, dim=-1))  # (b, N, F_out, T)


class cheb_conv(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])

    def forward(self, x):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)

            for k in range(self.K):

                T_k = self.cheb_polynomials[k]  # (N,N)

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = graph_signal.permute(0, 2, 1).matmul(T_k).permute(0, 2, 1)

                output = output + rhs.matmul(theta_k)

            outputs.append(output.unsqueeze(-1))

        return F.relu(torch.cat(outputs, dim=-1))

class Embedding(nn.Module):
    def __init__(self, nb_seq, d_Em, num_of_features, Etype):
        super(Embedding, self).__init__()
        self.nb_seq = nb_seq
        self.Etype = Etype
        self.num_of_features = num_of_features
        
        if Etype == 'T':
            # Temporal embedding
            self.pos_embed = nn.Embedding(nb_seq, d_Em)
            self.norm = nn.LayerNorm(d_Em)
        else:
            # Spatial embedding
            self.pos_embed = nn.Embedding(nb_seq, d_Em)
            self.norm = nn.LayerNorm(d_Em)

    def forward(self, x, batch_size):
        if self.Etype == 'T':
            # Input shape: (B, N, F, T)
            B, N, F, T = x.shape
            
            # Get positional embeddings
            pos = torch.arange(T, dtype=torch.long).to(x.device)
            pos_embed = self.pos_embed(pos)  # (T, d_Em)
            
            # Reshape for broadcasting: (1, 1, 1, T, d_Em)
            pos_embed = pos_embed.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            
            # Process input: (B, N, F, T) -> (B, N, F, T, 1)
            x = x.unsqueeze(-1)
            
            # Add with broadcasting
            embedding = x + pos_embed  # (B, N, F, T, d_Em)
            
            # Flatten for LayerNorm
            embedding = embedding.reshape(-1, self.pos_embed.embedding_dim)
            embedding = self.norm(embedding)
            
            # Reshape back: (B*N*F*T, d_Em) -> (B, N, F, T, d_Em) -> (B, N, d_Em, T)
            return embedding.reshape(B, N, F, T, -1).permute(0, 1, 4, 3, 2)[..., 0]
        else:
            # Spatial embedding
            # Input shape: (B, N, d_model)
            pos = torch.arange(x.size(1), dtype=torch.long).to(x.device)
            pos_embed = self.pos_embed(pos)  # (N, d_model)
            embedding = x + pos_embed.unsqueeze(0)  # (B, N, d_model)
            return self.norm(embedding)
            
class GTU(nn.Module):
    def __init__(self, in_channels, time_strides, kernel_size):
        super(GTU, self).__init__()
        self.in_channels = in_channels
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.con2out = nn.Conv2d(in_channels, 2 * in_channels, kernel_size=(1, kernel_size), stride=(1, time_strides))

    def forward(self, x):
        x_causal_conv = self.con2out(x)
        x_p = x_causal_conv[:, : self.in_channels, :, :]
        x_q = x_causal_conv[:, -self.in_channels:, :, :]
        x_gtu = torch.mul(self.tanh(x_p), self.sigmoid(x_q))
        return x_gtu



class DSTAGNN_block(nn.Module):
    def __init__(self, DEVICE, num_of_d, in_channels, K, nb_chev_filter, nb_time_filter, time_strides,
                 cheb_polynomials, adj_pa, adj_TMD, num_of_vertices, num_of_timesteps, d_model, d_k, d_v, n_heads):
        super(DSTAGNN_block, self).__init__()
        assert num_of_vertices == 2139, "Graph size mismatch"
        assert num_of_timesteps == 144, "Sequence length mismatch"      
        # Initialize with safe defaults
        self.d_model = min(d_model, 64)
        self.n_heads = min(n_heads, 2)
        self.d_k = min(d_k, 32)
        self.d_v = min(d_v, 32)
        self.num_of_vertices = num_of_vertices
        self.num_of_timesteps = num_of_timesteps
        # Properly register buffers for TPU
        self.register_buffer('adj_pa', torch.FloatTensor(adj_pa))
        self.register_buffer('adj_TMD', torch.FloatTensor(adj_TMD))
        
        # Temporal processing
        self.temporal_proj = nn.Sequential(
            nn.Linear(num_of_timesteps, self.d_model),
            nn.LayerNorm(self.d_model)
        )
        
        # Attention layers
        self.temporal_att = MultiHeadAttention(
            DEVICE, self.d_model, self.d_k, self.d_v, self.n_heads, num_of_d
        )
        self.spatial_att = SMultiHeadAttention(
            DEVICE, self.d_model, self.d_k, self.d_v, K
        )
        
        # Convolutional layers
        self.cheb_conv = cheb_conv_withSAt(
            K, cheb_polynomials, in_channels, 
            min(nb_chev_filter, 32),
            num_of_vertices
        )
        
        # GTU layers
        self.gtu3 = GTU(nb_time_filter, time_strides, 3)
        self.gtu5 = GTU(nb_time_filter, time_strides, 5) 
        self.gtu7 = GTU(nb_time_filter, time_strides, 7)
        
        # Residual connection
        self.res_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1,1), stride=(1,time_strides))
        self.ln = nn.LayerNorm(nb_time_filter)

    def forward(self, x, res_att):
        B, N, F, T = x.shape
        # assert num_of_timesteps == self.num_of_timesteps, \
        #     f"Input timesteps {num_of_timesteps} don't match configured {self.num_of_timesteps}"
        print(f"Input shape: {x.shape}")
        # assert x.shape[1] == self.num_of_vertices
        # assert x.shape[2] == num_of_features
        # assert x.shape[3] == self.num_of_timesteps
        print(f"Input device: {x.device}")
        print(f"Model device: {next(self.parameters()).device}")
        assert x.is_contiguous(), "Input not contiguous"
        print(f"Input stats - mean: {x.mean().item():.3f}, std: {x.std().item():.3f}")
                
    
    
        # Initialize res_att properly if it's the first block
        if isinstance(res_att, int):
            res_att = torch.zeros(
                batch_size, num_of_features, num_of_vertices, self.d_model,
                device=x.device, dtype=x.dtype
            )
        
        # 1. Temporal projection
        x_temp = x.permute(0,1,3,2)  # [B,N,T,F]
        x_temp = self.temporal_proj(x_temp.reshape(-1,T)).view(B,N,F,-1)
        
        # 2. Temporal attention
        temp_out, temp_att = self.temporal_att(
            x_temp, x_temp, x_temp,
            None, res_att
        )
        
        # 3. Spatial attention
        spatial_att = self.spatial_att(temp_out, temp_out, None)
        
        # 4. Chebyshev convolution
        x_conv = x.permute(0,2,1,3)  # [B,F,N,T]
        spatial_out = self.cheb_conv(x_conv, spatial_att, self.adj_pa)
        
        # 5. Temporal convolution
        time_out = torch.cat([
            self.gtu3(spatial_out),
            self.gtu5(spatial_out),
            self.gtu7(spatial_out)
        ], dim=-1)
        
        # 6. Residual connection
        res = self.res_conv(x.permute(0,2,1,3))
        output = F.relu(res + time_out)
        
        return output.permute(0,2,1,3), temp_att

class DSTAGNN_submodule(nn.Module):
    def __init__(self, DEVICE, num_of_d, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides,
                 cheb_polynomials, adj_pa, adj_TMD, num_for_predict, len_input, num_of_vertices, d_model, d_k, d_v, n_heads):
        super().__init__()
        
        # Safe initialization
        self.num_blocks = min(nb_block, 2)
        self.time_strides = time_strides
        
        # Initialize blocks
        self.blocks = nn.ModuleList()
        in_ch = in_channels
        for i in range(self.num_blocks):
            block = DSTAGNN_block(
                DEVICE, num_of_d, in_ch, K,
                nb_chev_filter, nb_time_filter,
                time_strides if i == 0 else 1,
                cheb_polynomials,
                adj_pa, adj_TMD, num_of_vertices,
                len_input//(time_strides**i),
                min(d_model,64), min(d_k,32), min(d_v,32), min(n_heads,2)
            )
            self.blocks.append(block)
            in_ch = nb_time_filter
        
        # Final projection
        self.final_proj = nn.Sequential(
            nn.Linear(nb_time_filter * self.num_blocks, 128),
            nn.Linear(128, num_for_predict)
        )

    def forward(self, x):
        B, N, F, T = x.shape
        outputs = []
        res_att = torch.zeros(B, F, N, 64, device=x.device)  # Adjusted for safe dims
        
        for block in self.blocks:
            x, res_att = block(x, res_att)
            outputs.append(x.mean(dim=-1))  # Reduce time dimension
            
        # Combine features
        features = torch.cat(outputs, dim=-1)
        return self.final_proj(features)
def make_model(DEVICE, num_of_d, nb_block, in_channels, K,
               nb_chev_filter, nb_time_filter, time_strides, adj_mx, adj_pa,
               adj_TMD, num_for_predict, len_input, num_of_vertices, d_model, d_k, d_v, n_heads):
    
    # Convert adjacency matrices to proper tensors
    if not torch.is_tensor(adj_mx):
        adj_mx = torch.FloatTensor(adj_mx).to(DEVICE)
    if not torch.is_tensor(adj_pa):
        adj_pa = torch.FloatTensor(adj_pa).to(DEVICE)
    if not torch.is_tensor(adj_TMD):
        adj_TMD = torch.FloatTensor(adj_TMD).to(DEVICE)
    
    # Compute Laplacian
    adj_mx_np = adj_mx.cpu().numpy() if torch.is_tensor(adj_mx) else adj_mx
    L_tilde = scaled_Laplacian(adj_mx_np)
    
    # Convert polynomials to device
    cheb_polynomials = [torch.from_numpy(i).float().to(DEVICE) for i in cheb_polynomial(L_tilde, K)]
    
    model = DSTAGNN_submodule(DEVICE, num_of_d, nb_block, in_channels,
                             K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials,
                             adj_pa, adj_TMD, num_for_predict, len_input, num_of_vertices, d_model, d_k, d_v, n_heads)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    return model
