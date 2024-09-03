import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim

import random
import pandas as pd
import numpy as np
device='cuda'
np.random.seed(1)
random.seed(1)
torch.manual_seed(1)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def attention(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        # q(batch_size, num_heads, seq_len, d_k)
        # k.transpose(-2, -1) (batch_size, num_heads, d_k, seq_len)
        # scores (batch_size, num_heads, seq_len, seq_len)
        # mask (batch_size, num_heads, seq_len, seq_len)
        
        
        if mask is not None:
            mask_expanded = mask.unsqueeze(1)
            scores = scores.masked_fill(mask_expanded == 0, -1e9)
        max_scores, _ = torch.max(scores, dim=-1, keepdim=True)
        scores = scores - max_scores
        scores = F.softmax(scores, dim=-1)
        output = torch.matmul(scores, v)
        
        return output

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # Linear projection and split into num_heads
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # (batch_size, num_heads, seq_len, d_k)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply attention on all the projected vectors
        attn_output = self.attention(q, k, v, mask)
        # attn_output: batch_size, num_heads, seq_len, d_k
        # Concatenate heads and put through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out(attn_output)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, dim_feedforward)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        src2 = self.self_attn(src, src, src, src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)  # layer norm
        src2 = self.feed_forward(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)  # layer norm
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src, mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, mask)
        return output


class Transformer(nn.Module):
    def __init__(self, input_fea_len, d_model, num_heads, num_layers, dim_feedforward):
        super(Transformer, self).__init__()
        self.embedding = nn.Linear(input_fea_len, d_model)
        encoder_layer = TransformerEncoderLayer(d_model, num_heads, dim_feedforward)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)
        self.query_W = nn.Linear(d_model, d_model)
        self.key_W = nn.Linear(d_model, d_model)
        self.value_W = nn.Linear(d_model, d_model)
        self.FC1 = nn.Linear(2, 1)
        self.FC2 = nn.Linear(d_model, 1)

    def forward(self, src, src_mask=None):
        src = self.embedding(src)
        output = self.encoder(src, src_mask)
        return output


class Critic_Attention(nn.Module):
    def __init__(self, d_model):
        super(Critic_Attention, self).__init__()
        self.query_W = nn.Linear(d_model, d_model)
        self.key_W = nn.Linear(d_model, d_model)
        self.value_W = nn.Linear(d_model, d_model)
        self.FC1 = nn.Linear(2, 1)

    # 1안
    def forward(self, query, key, value, mask=None):
        """
        Compute the scaled dot-product attention using PyTorch.

        Args:
        query: Tensor of shape (batch, 2, depth)
        key: Tensor of shape (batch, seq_len, depth)
        value: Tensor of shape (batch, seq_len, depth)
        mask: Tensor broadcastable to (batch, seq_len). Defaults to None.

        Returns:
        output: Tensor of shape (..., seq_len_q, depth_v)
        attention_weights: Tensor of shape (..., seq_len_q, seq_len_k)
        """
        batch, seq_len, fea = key.shape
        mask=1-mask
        query = self.query_W(query)
        key = self.key_W(key)
        value = self.value_W(value)

        query_1 = query.repeat(1, seq_len, 1)
        #query_2 = query[:, 1, :].unsqueeze(1).repeat(1, seq_len, 1)

        # 두 반복된 텐서를 concat하여 (batch, seq*2, 512)로 확장

        dot_product1 = torch.sum(key * query_1, dim=-1, keepdim=True)
        #dot_product2 = torch.sum(key * query_2, dim=-1, keepdim=True)

        scaled_attention_logits = dot_product1 / torch.sqrt(torch.tensor(fea, dtype=torch.float32))
        #scaled_dot_product2 = dot_product2 / torch.sqrt(torch.tensor(fea, dtype=torch.float32))
        #scaled_dot_product_cat = torch.cat((scaled_dot_product1, scaled_dot_product2), dim=2)  # (batch,seq,2)
        #scaled_attention_logits = self.FC1(scaled_dot_product1)  # (batch,seq,1)
        if mask is not None:
            scaled_attention_logits =scaled_attention_logits.masked_fill(mask == 1, 0) 
        
        state_value = torch.sum(scaled_attention_logits, dim=1)
        return state_value


class Actor_Attention(nn.Module):
    def __init__(self, d_model):
        super(Actor_Attention, self).__init__()
        self.query_W = nn.Linear(d_model, d_model)
        self.key_W = nn.Linear(d_model, d_model)
        self.value_W = nn.Linear(d_model, d_model)
        self.FC1 = nn.Linear(2, 1)
        self.FC2 = nn.Linear(d_model, 1)

    def forward(self, query, key, value, mask=None):
        """
        Compute the scaled dot-product attention using PyTorch.

        Args:
        query: Tensor of shape (batch, 2, depth)
        key: Tensor of shape (batch, seq_len, depth)
        value: Tensor of shape (batch, seq_len, depth)
        mask: Tensor broadcastable to (batch, seq_len). Defaults to None.

        Returns:
        output: Tensor of shape (..., seq_len_q, depth_v)
        attention_weights: Tensor of shape (..., seq_len_q, seq_len_k)
        """
        batch, seq_len, fea = key.shape

        query = self.query_W(query)
        key = self.key_W(key)
        value = self.value_W(value)
        mask=1-mask
        
        
        query_1 = query.repeat(1, seq_len, 1)
        
        #query_2 = query[:, 1, :].unsqueeze(1).repeat(1, seq_len, 1)

        # 두 반복된 텐서를 concat하여 (batch, seq*2, 512)로 확장
        
        dot_product1 = torch.sum(key * query_1, dim=-1, keepdim=True)
        #dot_product2 = torch.sum(key * query_2, dim=-1, keepdim=True)

        scaled_dot_product1 = dot_product1 / torch.sqrt(torch.tensor(fea, dtype=torch.float32))
        #scaled_dot_product2 = dot_product2 / torch.sqrt(torch.tensor(fea, dtype=torch.float32))
        #scaled_dot_product_cat = torch.cat((scaled_dot_product1, scaled_dot_product2), dim=2)
        #scaled_attention_logits = self.FC1(scaled_dot_product_cat)  # (batch,seq,1)
        scaled_attention_logits =scaled_dot_product1
        
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        attention_weights = F.softmax(scaled_attention_logits, dim=1)  # (batch,seq,1)
        output = attention_weights * value  # (batch,seq,1) , (batch,seq,dim)
        output = self.FC2(output)  # (batch,seq,1)
        
        if mask is not None:
            output += (mask * -1e9)
        output = F.softmax(output, dim=1)  # #(batch,seq,1)
        return output, attention_weights


class PPO(nn.Module):
    def __init__(self, learning_rate=0.001, clipping_ratio=0.2, machine_len=8, d_model=512, num_heads=16, num_layers=3,
                 dim_feedforward=1024):
        super(PPO, self).__init__()
        self.CA = Critic_Attention(d_model)
        self.AA = Actor_Attention(d_model)
        self.encoder = Transformer(machine_len+2, d_model, num_heads, num_layers, dim_feedforward)
        #self.decoder = Transformer(machine_fea_len, d_model, num_heads, num_layers, dim_feedforward)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.gamma = 1
        self.lmbda = 0.95
        self.epsilon=clipping_ratio
        self.alpha = 0.5
        self.linear1 = nn.Linear(machine_len, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.linear3 = nn.Linear(machine_len, dim_feedforward)
        self.linear4 = nn.Linear(dim_feedforward, d_model)
    
    def create_masking(self,mask):
        # mask의 크기는 (batch, seq)
        batch_size, seq_len = mask.size()

        # 1차원 마스크를 (seq, 1)로 확장하여 서로 곱할 수 있게 만듭니다.
        mask = mask.unsqueeze(2)  # 크기: (batch, seq, 1)

        # 마스크를 transpose하고 서로 곱해서 (seq, seq) 크기로 확장합니다.
        masking = mask * mask.transpose(1, 2)  # 크기: (batch, seq, seq)

        return masking

    def get_action(self, job_state, machine_state, mask_seq, ans=None):
        # mask_seq [1,0,....,1] (batch, seq) 1이 고려함
        # job state (batch, total_len, fea)
        # machine state (batch, total_len)
        mask = self.create_masking(mask_seq.clone())
        enh = self.encoder(job_state, mask)
        x = F.relu(self.linear1(machine_state))
        tnh = self.linear2(x).unsqueeze(1)
        
        output, attention_weights = self.AA(tnh, enh, enh, mask_seq.unsqueeze(2))  # (batch,seq,1)
        output = output.squeeze(-1)
        
        samples = torch.multinomial(output, 1)  # (B, 1) 크기의 인덱스 텐서
        if ans!=None:
            pi = output.squeeze(-1).gather(1, ans)
        else:
            pi = output.squeeze(-1).gather(1, samples)
        return samples, pi

    def calculate_v(self, job_state, machine_state, mask_seq):
        mask = self.create_masking(mask_seq.clone())
        enh = self.encoder(job_state, mask)
        x = F.relu(self.linear3(machine_state))
        tnh = self.linear4(x).unsqueeze(1)
        state_value = self.CA(tnh, enh, enh, mask_seq.unsqueeze(2))
        return state_value

    def update(self, episode, k_epoch,episode_num,step1,model_dir):

        job_states = [item[0] for item in episode]
        job_states = torch.stack(job_states, dim=0).squeeze(1).to(device)
        batch, seq, fea = job_states.shape
        machine_states = [item[1] for item in episode]
        machine_states = torch.stack(machine_states, dim=0).squeeze(1).to(device)
        actions = [item[2] for item in episode]
        actions = torch.tensor(actions).unsqueeze(1).to(device)
        pi_old = [item[3] for item in episode]
        pi_old = torch.tensor(pi_old).unsqueeze(1).to(device)
        rewards = [item[4] for item in episode]
        rewards = torch.tensor(rewards).unsqueeze(1).to(device)
        dones = [item[5] for item in episode]
        dones = torch.tensor(dones).unsqueeze(1).to(device)
        masks = [item[6] for item in episode]  # (1,seq)
        masks = torch.stack(masks, dim=0).squeeze(1).to(device)
        
        for _ in range(k_epoch):
            
            _,pi_new = self.get_action(job_states, machine_states,masks, actions)

            state_v = self.calculate_v(job_states, machine_states,masks)
            state_next_v = state_v[1:].clone()
            zero_row = torch.zeros(1, 1).to(device)
            state_next_v = torch.cat((state_next_v, zero_row), dim=0)
            td_target = rewards + self.gamma * state_next_v * dones
            delta = td_target - state_v
            advantage_lst = np.zeros(len(episode))
            advantage_lst = torch.tensor(advantage_lst, dtype=torch.float32).unsqueeze(1).to(device)
            job_num = int(len(episode) //episode_num)
            j = 0
                
            for i in range(episode_num):
                advantage = 0.0
                for t in reversed(range(j, j + job_num)):
                    
                    advantage = self.gamma * self.lmbda * advantage + delta[t][0]
                    advantage_lst[t][0] = advantage
                j += job_num
                
            
            ratio = torch.exp(torch.log(pi_new) - torch.log(pi_old))  # a/b == exp(log(a)-log(b))
            
            surr1 = ratio * advantage_lst
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage_lst
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(state_v, td_target.detach()) * self.alpha

            ave_loss = loss.mean().item()
            v_loss = (self.alpha * F.smooth_l1_loss(state_v, td_target.detach())).item()
            
            p_loss = -torch.min(surr1, surr2).mean().item()

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        if step1 % 10 == 0:
            torch.save({
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),

            }, model_dir+'trained_model' + str(step1) + '.pth')

        return ave_loss, v_loss, p_loss

