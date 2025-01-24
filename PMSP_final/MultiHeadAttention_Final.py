import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
import numpy as np
device='cuda'

class Attention(nn.Module):
    def __init__(self, d_model):
        super(Attention, self).__init__()
        
        self.d_model = d_model
        self.d_k = d_model

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def attention(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(2,1)) / math.sqrt(self.d_k)
        n_scores=scores.clone()
        # q (batch_size, seq_len, d_k)
        # k (batch_size, d_k, seq_len)
        # scores (batch_size, seq_len, seq_len)
        # mask (batch_size, seq_len, seq_len)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        tensor_max = torch.max(scores, dim=-1, keepdim=True).values
        scores = scores - tensor_max
        scores = F.softmax(scores, dim=-1)
        output = torch.matmul(scores, v)
        return output, scores,n_scores

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        # Linear projection and split into num_heads
        q = self.q_linear(q)
        # (batch_size, seq_len, d_k)
        k = self.k_linear(k)
        v = self.v_linear(v)

        # Apply attention on all the projected vectors
        attn_output,scores,n_scores = self.attention(q, k, v, mask)
        # attn_output: batch_size, seq_len, d_k
        # Concatenate heads and put through final linear layer
        return attn_output,scores,n_scores


class Transformer(nn.Module):
    def __init__(self, input_fea_len, d_model):
        super(Transformer, self).__init__()
        self.embedding = nn.Linear(input_fea_len, d_model)
        self.attention = Attention(d_model)
        

    def forward(self, state, state_mask=None):
        state_embedded = self.embedding(state)
        output,scores,n_scores = self.attention(state_embedded, state_embedded, state_embedded, state_mask)
        result = torch.cat((state_embedded, output), dim=2)
        return result,scores,n_scores

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, output_size)
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
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class PPO(nn.Module):
    def __init__(self, learning_rate=0.001, clipping_ratio=0.2, machine_len=10, d_model=512, num_heads=8, fea_len=25,num_layers=3,
                 dim_feedforward=1024):
        super(PPO, self).__init__()
        self.encoder2 = Transformer(fea_len, d_model)
        self.encoder1 = Transformer(fea_len, d_model)
        #self.decoder = Transformer(machine_fea_len, d_model, num_heads, num_layers, dim_feedforward)
        self.optimizer_policy = optim.Adam(self.parameters(), lr=learning_rate)
        self.optimizer_value = optim.Adam(self.parameters(), lr=learning_rate)
        self.gamma = 0.99
        self.lmbda = 0.95
        self.epsilon=clipping_ratio
        self.alpha = 0.5
        
        self.AA=MLP(d_model*2,1)
        self.CA=MLP(d_model*2,1)
        
        self.machine_len=machine_len
        


    def get_action(self, state, mask, action_mask, ans=None):
        # mask_seq [1,0,....,1] (batch, seq+m,seq+m) 1이 고려함
        # action_mask [1,0,....,1] (batch, seq) 1이 고려함
        # state (batch, n+m, fea)
        batch,seq,fea=state.shape
        
        enh,att_score,n_scores = self.encoder1(state, mask) #(batch,n+m,fea)
        tensor = enh[:, :-self.machine_len, :]
        #mask masking=0
        if ans == None:
        
            output=self.AA(tensor) #(batch,n,1)

            if action_mask is not None:
                mask_expanded = action_mask.unsqueeze(2)
                output = output.masked_fill(mask_expanded == 0, -1e9)
                

            
            tensor_max = torch.max(output, dim=1, keepdim=True).values
            output = output- tensor_max
            output = F.softmax(output, dim=1)
            output = output.squeeze(-1)
            samples = torch.argmax(output, dim=1, keepdim=True)
            
            #samples = torch.multinomial(output, 1)  # (B, 1) 크기의 인덱스 텐서
            pi = output.squeeze(-1).gather(1, samples)

        elif ans!=None:
            output = self.AA(tensor)  # (batch,n,1)

            if mask is not None:
                mask_expanded = mask_seq.unsqueeze(2)
                output = output.masked_fill(mask_expanded == 0, -1e9)

            tensor_max = torch.max(output, dim=1, keepdim=True).values
            output = output - tensor_max
            output = F.softmax(output, dim=1)
            output = output.squeeze(-1)
            samples=0
            pi = output.squeeze(-1).gather(1, ans)

        return samples, pi,enh,att_score,n_scores

    def calculate_v(self, state, mask_seq):
        # mask_seq [1,0,....,1] (batch, seq) 1이 고려함
        
        batch,seq,fea=state.shape
        
        machine_mask = torch.ones((batch, self.machine_len)).to(device)  # 예: (3, 2) 크기의 텐서
        total_mask = torch.cat((mask_seq.clone(), machine_mask), dim=1)
        
        mask = self.create_masking(total_mask)
        enh,att_score = self.encoder2(state, mask) #(batch,n+m,fea)
        fea=enh.shape[2]
        if mask is not None:
            mask_expanded = total_mask.unsqueeze(2).repeat(1,1,fea)
            
            enh ==enh.masked_fill(mask_expanded == 0, 0) 
        enh=torch.mean(enh, dim=1, keepdim=True) 
        state_value=self.CA(enh).squeeze(-1) #(batch,1,fea)
        
        return state_value

    def update(self, episode,job_len, k_epoch,step1,model_dir):

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
            
            _,pi_new,_,_ = self.get_action(states, masks, actions)
            state_v = self.calculate_v(states, masks)
            
            state_next_v = state_v[1:].clone()
            zero_row = torch.zeros(1, 1).to(device)

            state_next_v = torch.cat((state_next_v, zero_row), dim=0)
            
            td_target = rewards + self.gamma * state_next_v * dones
            delta = td_target - state_v
            advantage_lst = np.zeros(len(episode))
            advantage_lst = torch.tensor(advantage_lst, dtype=torch.float32).unsqueeze(1).to(device)
            episode_num = int(len(episode) //job_len)
            j = 0
                
            for i in range(episode_num):
                advantage = 0.0
                for t in reversed(range(j, j + job_len)):
                    
                    advantage = self.gamma * self.lmbda * advantage + delta[t][0]
                    advantage_lst[t][0] = advantage
                    
                j += job_len
                
            
            ratio = torch.exp(torch.log(pi_new) - torch.log(pi_old))  # a/b == exp(log(a)-log(b))
            surr1 = ratio * advantage_lst
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage_lst
            # Policy loss
            loss_policy = -torch.min(surr1, surr2).mean()
            p_loss = -torch.min(surr1, surr2).mean().item()
            self.optimizer_policy.zero_grad()
            loss_policy.backward()
            self.optimizer_policy.step()
            state_v = self.calculate_v(states, masks)
            
            state_next_v = state_v[1:].clone()
            zero_row = torch.zeros(1, 1).to(device)

            state_next_v = torch.cat((state_next_v, zero_row), dim=0)
            
            td_target = rewards + self.gamma * state_next_v * dones
            # Value loss
            loss_value = F.smooth_l1_loss(state_v, td_target.detach())
            v_loss = (F.smooth_l1_loss(state_v, td_target.detach())).item()
            self.optimizer_value.zero_grad()
            loss_value.backward()
            self.optimizer_value.step()

            
            
        
            ave_loss = p_loss+v_loss
            
            
        
        if step1 % 40 == 0:
            torch.save({
                'model_state_dict': self.state_dict(),
                'optimizer_state_v_dict': self.optimizer_value.state_dict(),
                'optimizer_state_p_dict': self.optimizer_policy.state_dict(),
            }, model_dir+'trained_model' + str(step1) + '.pth')

        return ave_loss, v_loss, p_loss

