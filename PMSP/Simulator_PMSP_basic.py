import random
import pandas as pd
import heapq
import numpy as np
from MultiHeadAttention_basic import *
random.seed(42)
device='cuda'
import torch
class PMSPScheduler:
    def __init__(self, num_machines=12, initial_jobs=80, additional_jobs=10, additional_arrivals=5, 
                 processing_time_range=(5, 15), machine_speed=[1,1,1,1,1,1,1.5,1.5,1.5,1.5,1.5,1.5], start_additional_arrival=20, arrival_interval=10,setup_range=(5,15),family_setup_num=10):
        self.num_machines = num_machines
        self.initial_jobs = initial_jobs
        self.additional_jobs = additional_jobs
        self.additional_arrivals = additional_arrivals
        self.processing_time_range = processing_time_range
        self.start_additional_arrival = start_additional_arrival
        self.arrival_interval = arrival_interval
        self.MP=float(10*(initial_jobs+additional_jobs*additional_arrivals)/num_machines+10*(8+130)/2/12)
        self.tardiness_factor=0.1
        self.duedate_range=0.5
        self.schedule = []
        self.machine_speed=machine_speed
        self.setup_range=setup_range
        self.setup_num=family_setup_num
        A = np.random.uniform(setup_range[0], setup_range[1], (family_setup_num, family_setup_num))
        symmetric_A = np.triu(A) + np.triu(A, 1).T
        np.fill_diagonal(symmetric_A, 0)
        self.setup=symmetric_A
        
    def generate_jobs(self):
        jobs = np.empty((0, 8))
        job_count=0
        for i in range(self.initial_jobs):
            job_id = job_count
            arrival_time = 0
            processing_time = random.randint(*self.processing_time_range)
            tardy_time=np.random.uniform(max(arrival_time+processing_time+10,self.MP*(1-self.tardiness_factor-self.duedate_range/2)),self.MP*(1-self.tardiness_factor+self.duedate_range/2))
            setup_type=np.random.randint(0, self.setup_num)
            property_j=[job_id,arrival_time,processing_time,setup_type,tardy_time,0.0,1.0, 0.0]
            jobs = np.vstack([jobs, np.array(property_j)])
            job_count+=1

        for j in range(self.additional_arrivals):
            arrival_time = self.start_additional_arrival + j * self.arrival_interval
            for k in range(self.additional_jobs):
                job_id = job_count
                processing_time = random.randint(*self.processing_time_range)
                tardy_time=np.random.uniform(max(arrival_time+processing_time+10,self.MP*(1-self.tardiness_factor-self.duedate_range/2)),self.MP*(1-self.tardiness_factor+self.duedate_range/2))
                setup_type=np.random.randint(0, self.setup_num)
                property_j=[job_id,arrival_time,processing_time,setup_type,tardy_time,0.0,0.0, 0.0]
                jobs = np.vstack([jobs, np.array(property_j)])
                job_count+=1
        return jobs
    
    def get_agent(self, machine_end_time):
        # 리스트에서 최소값을 찾습니다.
        min_value = min(machine_end_time)
        # 최소값의 모든 인덱스를 찾습니다.
        min_indices = [i for i, x in enumerate(machine_end_time) if x == min_value]
        # 최소값의 인덱스 중 랜덤하게 하나를 선택합니다.
        chosen_index = random.choice(min_indices)
        return min_value, chosen_index
    def calculate_tardy(self, job_u_s,current_time):
        
        # job_id / arrival_time / processing_time / familiy_type / tardy_time / tardy_occur /mask / processed 여부
        tardy_occured = job_u_s[:,-3].sum()
        not_processed=job_u_s[job_u_s[:,-1]==0].copy()
        tardy_yet=np.maximum(not_processed[:,2]+current_time-not_processed[:,-3],0).sum()
        return tardy_occured+tardy_yet
 
    
    def schedule_jobs(self,jobs,episode,ppo):
        machines = []
        jobss = []
        start_times = []  # start times as floats
        durations = []      # durations as floats
        total_tardy=0
        total_reward=0
        
        past_tardy=0
        # 각 머신의 작업 큐를 저장할 리스트 (우선순위 큐로 구현)
        machine_matrix=np.zeros((self.num_machines,3))
        machine_matrix[:,2]=self.machine_speed
        # remainning processing time, current setup, speed
        current_time=0.0
        # job_id / arrival_time / processing_time / familiy_type / tardy_time / tardy_occur /mask / processed 여부
        job_len=jobs.shape[0]
        # job_id / arrival_time / processing_time / familiy_type / tardy_time / tardy_occur /mask / processed 여부
        jobs_u_s=jobs.copy()
        state=np.zeros((job_len+self.num_machines,6))
        
        for i in range(job_len):
            min_value, chosen_index=self.get_agent(machine_matrix[:,0].copy()) # batch, n+m, fea
            current_time+=min_value
            machine_matrix[:,0]=machine_matrix[:,0]-min_value
            # 현재 시간보다 작고 processing이 되지 않은 job의 마스킹 변경, 
            condition = (jobs_u_s[:, -1] == 0) & (jobs_u_s[:, 1] <=current_time)
            # 조건을 만족하는 행의 1번째 열 값을 1로 변경
            jobs_u_s[condition, -2] = 1
            
            count = np.sum(jobs_u_s[:, -2] == 1)
            while count==0:
                filtered_arr = jobs_u_s[jobs_u_s[:, -1] == 0]
                min_row = np.min(filtered_arr[:, 1])
                gap=min_row-current_time
                current_time=min_row
                machine_matrix[:,0]=machine_matrix[:,0]-gap
                machine_matrix[machine_matrix[:, 0] < 0, 0] = 0
                
                condition = (jobs_u_s[:, -1] == 0) & (jobs_u_s[:, 1] <=current_time)
                # 조건을 만족하는 행의 1번째 열 값을 1로 변경
                jobs_u_s[condition, -2] = 1
                count = np.sum(jobs_u_s[:, -2] == 1)

            mask=torch.tensor(jobs_u_s[:,-2].copy(),dtype=torch.float32).unsqueeze(0).to(device) #(1, seq)
            # job_id / arrival_time / processing_time / familiy_type / tardy_time / tardy_occur /mask / processed 여부
            job_state=jobs_u_s[:,2:-3].copy() #processing / family type / tardy
            job_state[:,-1]=job_state[:,-1]-current_time
            state[:job_len,:3]=job_state
            machine_pro = machine_matrix[chosen_index,:]
            
            # (6,3) 크기로 행을 복제
            machine_pro = np.tile(machine_pro, (job_len, 1))
            state[:job_len,3:]=machine_pro
            state[job_len:,3:]=machine_matrix

            
            state_tensor=torch.tensor(state.copy(),dtype=torch.float32).unsqueeze(0).to(device)/20.0 # batch, n+m, fea
            
            action,pi=ppo.get_action(state_tensor,mask,ans=None)
            job_index=action.item()
            # job_id / arrival_time / processing_time / familiy_type / tardy_time / tardy_occur /mask / processed 여부
            machine_matrix[chosen_index,0]=jobs_u_s[job_index][2]/machine_matrix[chosen_index,2]+self.setup[int(machine_matrix[chosen_index,1])][int(jobs_u_s[job_index][3])]
            machine_matrix[chosen_index,1]=jobs_u_s[job_index][3]
            jobs_u_s[job_index,-2]=0
            jobs_u_s[job_index,-1]=1

            total_tardy+=max((current_time+machine_matrix[chosen_index,0])-jobs_u_s[job_index][-4],0)
            # job_id / arrival_time / processing_time / familiy_type / tardy_time / tardy_occur /mask / processed 여부
            jobs_u_s[job_index][-3]=max((current_time+machine_matrix[chosen_index,0])-jobs_u_s[job_index][-4],0)
            tardy=self.calculate_tardy(jobs_u_s,current_time)
            reward=past_tardy-tardy
            total_reward+=reward
            past_tardy=tardy
            
           
            #episode.append([job_state.clone(), machine_state.clone(), action, pi, reward, dones, mask, job_next_state.clone(), machine_next_state.clone()])
            if i==jobs.shape[0]-1:
                done=0
            else:
                done=1
            
            episode.append([state_tensor.clone(), action.item(), pi.item(), reward, done, mask.clone()])
            # job_id / arrival_time / processing_time / familiy_type / tardy_time / tardy_occur /mask / processed 여부
            machines.append("Machine "+str(chosen_index))
            jobss.append("Job "+str(job_index))
            start_times.append(current_time)
            durations.append(jobs_u_s[job_index][2])
            
        return machines,jobss,start_times,durations,episode,total_tardy,total_reward
     
    def plot_gantt(self,machines,jobs,start_times,durations):
                # Prepare data for Gantt chart
        df = pd.DataFrame({
            'Machine': machines,
            'Job': jobs,
            'Start': start_times,
            'Finish': [start_times[i] + durations[i] for i in range(len(start_times))]
        })
        
        # Create Gantt chart
        fig, ax = plt.subplots(figsize=(10, 6))

        for i, row in df.iterrows():
            ax.barh(row['Machine'], row['Finish'] - row['Start'], left=row['Start'], label=row['Job'])

        # Add labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel('Machine')
        ax.set_title('Gantt Chart')

        # Optional: Add job labels inside the bars
        for i, row in df.iterrows():
            ax.text(row['Start'] + (row['Finish'] - row['Start']) / 2, row['Machine'], row['Job'],
                    va='center', ha='center', color='white', fontweight='bold')

        plt.show()
    
    def run_simulation(self,cat_num,cat_num2,ppo):
        episode=[]
        ave_tardy=[]
        for __ in range(cat_num):
            jobs=self.generate_jobs()
            jobs[:,1:]=jobs[:,1:].astype(float)
            for _ in range(cat_num2):  
                machines,jobss,start_times,durations,episode,total_tardy,total_reward=self.schedule_jobs(jobs,episode,ppo)
                ave_tardy.append(total_tardy)
            #self.plot_gantt(machines,jobss,start_times,durations)
        ave_t=np.array(ave_tardy).mean()
        return ave_t,ave_tardy,episode
# 결과를 저장할 경우
    
# schedule_df.to_csv("schedule_results.csv", index=False)
