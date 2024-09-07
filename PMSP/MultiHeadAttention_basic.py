import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
import numpy as np
device='cuda'

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
        scores = F.softmax(scores, dim=-1)
        tensor_max = torch.max(scores, dim=-1, keepdim=True).values
        scores = scores - tensor_max
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
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
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
        

    def forward(self, src, src_mask=None):
        src = self.embedding(src)
        output = self.encoder(src, src_mask)
        return output

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_size)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # He 초기화를 사용하여 가중치를 초기화합니다.
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                # 편향을 0으로 초기화합니다.
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                # 배치 정규화 레이어의 가중치를 초기화합니다.
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class PPO(nn.Module):
    def __init__(self, learning_rate=0.001, clipping_ratio=0.2, job_len=130, machine_len=12, d_model=512, num_heads=16, fea_len=6,num_layers=3,
                 dim_feedforward=1024):
        super(PPO, self).__init__()
        self.encoder2 = Transformer(fea_len, d_model, num_heads, num_layers, dim_feedforward)
        self.encoder1 = Transformer(fea_len, d_model, num_heads, num_layers, dim_feedforward)
        #self.decoder = Transformer(machine_fea_len, d_model, num_heads, num_layers, dim_feedforward)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.gamma = 0.99
        self.lmbda = 0.95
        self.epsilon=clipping_ratio
        self.alpha = 0.5
        
        self.AA=MLP(d_model,1)
        self.CA=MLP(d_model,1)
        self.job_len=job_len
        self.machine_len=machine_len
        
    def create_masking(self,mask):
        # mask의 크기는 (batch, seq)
        batch_size, seq_len = mask.size()

        # 1차원 마스크를 (seq, 1)로 확장하여 서로 곱할 수 있게 만듭니다.
        mask = mask.unsqueeze(2)  # 크기: (batch, seq, 1)

        # 마스크를 transpose하고 서로 곱해서 (seq, seq) 크기로 확장합니다.
        masking = mask * mask.transpose(1, 2)  # 크기: (batch, seq, seq)

        return masking

    def get_action(self, state, mask_seq, ans=None):
        # mask_seq [1,0,....,1] (batch, seq) 1이 고려함
        # state (batch, n+m, fea)
        batch,seq,fea=state.shape
        machine_mask = torch.ones((batch, self.machine_len)).to(device)  # 예: (3, 2) 크기의 텐서
        total_mask = torch.cat((mask_seq.clone(), machine_mask), dim=1)
        mask = self.create_masking(total_mask)
        enh = self.encoder1(state, mask) #(batch,n+m,fea)
        tensor = enh[:, :self.job_len, :]
        #mask masking=0
        
        output=self.AA(tensor) #(batch,n,1)
        
        if mask is not None:
            mask_expanded = mask_seq.unsqueeze(2)
            output = output.masked_fill(mask_expanded == 0, -1e9)
        
        tensor_max = torch.max(output, dim=1, keepdim=True).values
        output = output- tensor_max
        output = F.softmax(output, dim=1)
        output = output.squeeze(-1)
        
        samples = torch.multinomial(output, 1)  # (B, 1) 크기의 인덱스 텐서
        if ans!=None:
            pi = output.squeeze(-1).gather(1, ans)
        else:
            pi = output.squeeze(-1).gather(1, samples)
        
        return samples, pi

    def calculate_v(self, state, mask_seq):
        # mask_seq [1,0,....,1] (batch, seq) 1이 고려함
        
        batch,seq,fea=state.shape
        
        machine_mask = torch.ones((batch, self.machine_len)).to(device)  # 예: (3, 2) 크기의 텐서
        total_mask = torch.cat((mask_seq.clone(), machine_mask), dim=1)
        
        mask = self.create_masking(total_mask)
        enh = self.encoder2(state, mask) #(batch,n+m,fea)
        fea=enh.shape[2]
        if mask is not None:
            mask_expanded = total_mask.unsqueeze(2).repeat(1,1,fea)
            
            enh ==enh.masked_fill(mask_expanded == 0, 0) 
        enh=torch.mean(enh, dim=1, keepdim=True) 
        state_value=self.CA(enh).squeeze(-1) #(batch,1,fea)
        
        return state_value

    def update(self, episode, k_epoch,step1,model_dir):

        states = [item[0] for item in episode]
        states = torch.stack(states, dim=0).squeeze(1).to(device)
        
        batch, seq, fea = states.shape
        actions = [item[1] for item in episode]
        actions = torch.tensor(actions).unsqueeze(1).to(device)
        pi_old = [item[2] for item in episode]
        pi_old = torch.tensor(pi_old).unsqueeze(1).to(device)
        rewards = [item[3] for item in episode]
        rewards = torch.tensor(rewards).unsqueeze(1).to(device)
        dones = [item[4] for item in episode]
        dones = torch.tensor(dones).unsqueeze(1).to(device)
        masks = [item[5] for item in episode]  # (1,seq)
        masks = torch.stack(masks, dim=0).squeeze(1).to(device)
        
        for _ in range(k_epoch):
            
            _,pi_new = self.get_action(states, masks, actions)
            state_v = self.calculate_v(states, masks)
            
            state_next_v = state_v[1:].clone()
            zero_row = torch.zeros(1, 1).to(device)

            state_next_v = torch.cat((state_next_v, zero_row), dim=0)
            
            td_target = rewards + self.gamma * state_next_v * dones
            delta = td_target - state_v
            advantage_lst = np.zeros(len(episode))
            advantage_lst = torch.tensor(advantage_lst, dtype=torch.float32).unsqueeze(1).to(device)
            episode_num = int(len(episode) //self.job_len)
            j = 0
                
            for i in range(episode_num):
                advantage = 0.0
                for t in reversed(range(j, j + self.job_len)):
                    
                    advantage = self.gamma * self.lmbda * advantage + delta[t][0]
                    advantage_lst[t][0] = advantage
                    
                j += self.job_len
                
            
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
        
        if step1 % 100 == 0:
            torch.save({
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),

            }, model_dir+'trained_model' + str(step1) + '.pth')

        return ave_loss, v_loss, p_loss

