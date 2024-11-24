import random
import pandas as pd
import heapq
import numpy as np
import matplotlib.pyplot as plt
from MultiHeadAttention_basic import *
random.seed(42)
import seaborn as sns
import torch
device='cuda'
# 원본 논문 세팅 PMSPScheduler(12,80,10,5,(5,15),[1,1,1,1,1,1,1.5,1.5,1.5,1.5,1.5,1.5],0,20,10,(5,15),10)
# Proposed setting PMSPScheduler(10,100,20,20,(10,20),[1,1,1,1,1,1.5,1.5,1.5,1.5,1.5],1,50,50,(5,15),10)
# Proposed setting PMSPScheduler(10,0,20,25,(10,20),[1,1,1,1,1,1.5,1.5,1.5,1.5,1.5],1,40,40,(5,15),10)
# Proposed setting PMSPScheduler(10,200,20,15,(10,20),[1,1,1,1,1,1.5,1.5,1.5,1.5,1.5],1,60,60,(5,15),10)


class PMSPScheduler:
    def __init__(self, num_machines=16, initial_jobs=200, additional_jobs=10, additional_arrivals=6, 
                 processing_time_range=(5, 15), machine_speed=[1,1,1,1,1,1,1,1,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25], sim_type=0, start_additional_arrival=20, arrival_interval=10,setup_range=(5,15),family_setup_num=6):
        # 머신 개수
        self.num_machines = num_machines
        # 초기 작업
        self.initial_jobs = initial_jobs
        # 한번에 도착하는 작업의 수
        self.additional_jobs = additional_jobs
        # 도착 횟수
        self.additional_arrivals = additional_arrivals
        # processing time range
        self.processing_time_range = processing_time_range
        # 도착을 시작하는 시간
        self.start_additional_arrival = start_additional_arrival
        # 도착 사이의 간격
        self.arrival_interval = arrival_interval
        self.setup_range=setup_range
        self.setup_num=family_setup_num
        # 원본 논문 세팅
        
        self.tardiness_factor=0.2
        self.duedate_range=0.2
        self.K=2
        self.sim_type=sim_type
        self.schedule = []
        self.machine_speed=machine_speed
        self.MP=float((self.processing_time_range[0]+self.processing_time_range[1])*(initial_jobs+additional_jobs*additional_arrivals)/num_machines/2+10*(self.setup_num+initial_jobs+additional_jobs*additional_arrivals)/2/self.num_machines)
        if self.sim_type==0:
            print('MP: ',self.MP)

        
    def generate_jobs(self):
        jobs = np.empty((0, 8))
        job_count=0
        A = np.random.uniform(self.setup_range[0], self.setup_range[1], (self.setup_num, self.setup_num))
        symmetric_A = np.triu(A) + np.triu(A, 1).T
        np.fill_diagonal(symmetric_A, 0)
        setup = symmetric_A
        if self.sim_type==0 or self.sim_type==1:
            for i in range(self.initial_jobs):
                job_id = job_count
                arrival_time = 0
                processing_time = random.uniform(*self.processing_time_range)
                if self.sim_type==0:
                    tardy_time=np.random.uniform(max(arrival_time+processing_time+10,self.MP*(1-self.tardiness_factor-self.duedate_range/2)),self.MP*(1-self.tardiness_factor+self.duedate_range/2))
                if self.sim_type==1:
                    tardy_time=arrival_time+self.K*np.random.uniform(0.5,1.5)*((self.processing_time_range[0]+self.processing_time_range[1])/2+(self.setup_range[0]+self.setup_range[1])/2)
                setup_type=np.random.randint(0, self.setup_num)
                property_j=[job_id,arrival_time,processing_time,setup_type,tardy_time,0.0,1.0, 0.0]
                jobs = np.vstack([jobs, np.array(property_j)])
                job_count+=1

            for j in range(self.additional_arrivals):
                arrival_time = self.start_additional_arrival + j * self.arrival_interval

                for k in range(self.additional_jobs):
                    job_id = job_count
                    processing_time = random.uniform(*self.processing_time_range)
                    if self.sim_type==0:
                        tardy_time=np.random.uniform(max(arrival_time+processing_time+10,self.MP*(1-self.tardiness_factor-self.duedate_range/2)),self.MP*(1-self.tardiness_factor+self.duedate_range/2))
                    if self.sim_type==1:
                        tardy_time=arrival_time+self.K*np.random.uniform(0.5,1.5)*((self.processing_time_range[0]+self.processing_time_range[1])/2+(self.setup_range[0]+self.setup_range[1])/2)
                    setup_type=np.random.randint(0, self.setup_num)
                    property_j=[job_id,arrival_time,processing_time,setup_type,tardy_time,0.0,0.0, 0.0]
                    jobs = np.vstack([jobs, np.array(property_j)])
                    job_count+=1

        if self.sim_type==2:
            arrival_time_type=np.zeros(self.setup_num)
            for i in range(self.initial_jobs):
                job_id = job_count
                arrival_time = 0
                processing_time = random.randint(*self.processing_time_range)
                tardy_time =self.K*(processing_time+(self.setup_range[0]+self.setup_range[1])/2)
                setup_type = job_count%self.setup_num
                property_j = [job_id, arrival_time, processing_time, setup_type, tardy_time, 0.0, 1.0, 0.0]
                jobs = np.vstack([jobs, np.array(property_j)])
                job_count += 1

            for i in range(self.additional_arrivals):
                inter_arrival_time=np.random.exponential(4*self.num_machines/(self.processing_time_range[0]+self.processing_time_range[1])/self.setup_num)
                setup_type = job_count % self.setup_num
                job_id = job_count
                arrival_time_type[setup_type]=arrival_time_type[setup_type]+inter_arrival_time
                arrival_time=arrival_time_type[setup_type]
                processing_time = random.randint(*self.processing_time_range)
                tardy_time=arrival_time+self.K*(processing_time+(self.setup_range[0]+self.setup_range[1])/2)
                property_j = [job_id, arrival_time, processing_time, setup_type, tardy_time, 0.0, 0.0, 0.0]
                jobs = np.vstack([jobs, np.array(property_j)])
                job_count += 1
        # jobs
        # job_id / arrival_time / processing_time / familiy_type / tardy_time / tardy_occur /mask / processed 여부
        return jobs, setup

    def show_ideal_queue_graph(self,jobs,time_horizon):
        queue_x=np.linspace(0,time_horizon,time_horizon*10)
        queue=np.zeros(int(time_horizon*10))
        for e,i in enumerate(queue_x):
            count = (jobs[:, 1] <= i).sum()
            queue[e]=max(count-int(self.num_machines*np.array(self.machine_speed).mean()*i/((self.processing_time_range[0]+self.processing_time_range[1])/2+(self.setup_range[0]+self.setup_range[1])/2)),0)
        plt.plot(queue_x,queue)
        plt.xlabel('Time', fontsize=14)
        plt.ylabel('Jobs in queue', fontsize=14)
        plt.title('Ideal queue graph', fontsize=16)
        plt.show()

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
        tardy_yet=np.maximum(not_processed[:,2]+current_time-not_processed[:,-4],0).sum()
        return tardy_occured+tardy_yet
 
    
    def schedule_jobs(self,jobs,setup,episode,ppo,mod):
        machines = []
        jobss = []
        start_times = []  # start times as floats
        durations = []      # durations as floats
        total_tardy=0
        total_reward=0
        
        past_tardy=0
        # 각 머신의 작업 큐를 저장할 리스트 (우선순위 큐로 구현)
        machine_matrix=np.zeros((self.num_machines,3))
        machine_matrix[:,1]=self.machine_speed
        # remainning processing time,  speed, current setup
        
        current_time=0.0
        # job_id / arrival_time / processing_time / familiy_type / tardy_time / tardy_occur /mask / processed 여부
        #mask가 1이면 사용 가능한 job
        job_len=jobs.shape[0]
        # job_id / arrival_time / processing_time / familiy_type / tardy_time / tardy_occur /mask / processed 여부
        jobs_u_s=jobs.copy()
        # state / processing time / setup /
        state=np.zeros((job_len+self.num_machines,5+self.setup_num*2))
        
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


            state[:job_len,[0,1]]=jobs_u_s[:,[2,4]].copy()
            state[:job_len,1]=state[:job_len,1]-current_time
            

            machine_pro = machine_matrix[chosen_index,:]
            
            # (6,3) 크기로 행을 복제
            machine_pro = np.tile(machine_pro, (job_len, 1))
            state[:job_len,3+self.setup_num:6+self.setup_num]=machine_pro.copy()
            state[job_len:,3+self.setup_num:6+self.setup_num]=machine_matrix.copy()
            current_machine_setup=machine_matrix[chosen_index,2]
            for i in range(job_len):
                state[i, 3 + int(jobs_u_s[i, 3])] = 1
                state[i, 2] = setup[int(machine_matrix[chosen_index, 2])][int(jobs_u_s[i][3])]
                state[i,5+self.setup_num]=0
                state[i,5+self.setup_num+int(current_machine_setup)]=1
            for i in range(self.num_machines):
                machine_setup=state[i+job_len,5+self.setup_num]
                state[i+job_len,5+self.setup_num]=0
                state[i+job_len,5+self.setup_num+int(machine_setup)]=1
            
            #print(pd.DataFrame(state))
            state_tensor=torch.tensor(state.copy(),dtype=torch.float32).unsqueeze(0).to(device)/100.0 # batch, n+m, fea
            if mod=='RL':
                action,pi=ppo.get_action(state_tensor,mask,ans=None)

            if mod=='SSPT':
                rows_with_one = (jobs_u_s[:, -2] == 1).nonzero()[0]
                z = np.zeros(len(rows_with_one))

                for i in range(len(rows_with_one)):
                    n = rows_with_one[i]
                    # job_id / arrival_time / processing_time / familiy_type / tardy_time / tardy_occur /mask / processed 여부
                    # remainning processing time,  speed current setup
                    z[i] = jobs_u_s[n][2]/machine_matrix[chosen_index,1]+setup[int(machine_matrix[chosen_index,2])][int(jobs_u_s[n][3])]

                min_value = np.min(z)

                # 최소값의 인덱스들 찾기
                min_indices = np.where(z == min_value)[0]
                # 최소값 중 무작위로 하나 선택
                min_index= np.random.choice(min_indices)
                min_index = rows_with_one[min_index]
                action = min_index

            if mod == 'ATCS':
                rows_with_one = (jobs_u_s[:, -2] == 1).nonzero()[0]
                pt_average = np.zeros(len(rows_with_one))
                st_average = np.zeros(len(rows_with_one))
                z = np.zeros(len(rows_with_one))

                for i in range(len(rows_with_one)):
                    n = rows_with_one[i]
                    # job_id / arrival_time / processing_time / familiy_type / tardy_time / tardy_occur /mask / processed 여부
                    # remainning processing time, current setup, speed
                    pt_average[i] = jobs_u_s[n][2]/machine_matrix[chosen_index,1]
                    st_average[i] = setup[int(machine_matrix[chosen_index,2])][int(jobs_u_s[n][3])]
                pt_a = pt_average.mean()+0.01
                st_a = st_average.mean()+0.01
                for i in range(len(rows_with_one)):
                    n = rows_with_one[i]
                    # job_id / arrival_time / processing_time / familiy_type / tardy_time / tardy_occur /mask / processed 여부
                    # remainning processing time, speed , current setup
                    st = setup[int(machine_matrix[chosen_index,2])][int(jobs_u_s[n][3])]
                    pt= jobs_u_s[n][2]/machine_matrix[chosen_index,1]                    
                    z[i] = np.log(1/pt*math.exp(-max(current_time+st+pt-jobs_u_s[n][4],0)/pt_a)*math.exp(-st/st_a))
                min_value = np.max(z)

                # 최소값의 인덱스들 찾기
                min_indices = np.where(z == min_value)[0]
                # 최소값 중 무작위로 하나 선택
                
                min_index = np.random.choice(min_indices)
                
                
                min_index = rows_with_one[min_index]
                action = min_index

            if mod == 'MDD':
                rows_with_one = (jobs_u_s[:, -2] == 1).nonzero()[0]
                z = np.zeros(len(rows_with_one))
                for i in range(len(rows_with_one)):
                    n = rows_with_one[i]
                    # job_id / arrival_time / processing_time / familiy_type / tardy_time / tardy_occur /mask / processed 여부
                    # remainning processing time, speed, current setup
                    z[i] = max(jobs_u_s[n][4],jobs_u_s[n][2]+current_time)
                min_index = np.argmin(z)
                min_index = rows_with_one[min_index]
                action = min_index
            if mod == 'FIFO':
                rows_with_one = (jobs_u_s[:, -2] == 1).nonzero()[0]
                z = np.zeros(len(rows_with_one))
                for i in range(len(rows_with_one)):
                    n = rows_with_one[i]
                    # job_id / arrival_time / processing_time / familiy_type / tardy_time / tardy_occur /mask / processed 여부
                    # remainning processing time, speed, current setup
                    z[i] = jobs_u_s[n][1]
                min_value = np.min(z)

                # 최소값의 인덱스들 찾기
                min_indices = np.where(z == min_value)[0]
                # 최소값 중 무작위로 하나 선택
                min_index = np.random.choice(min_indices)
                min_index = rows_with_one[min_index]
                action = min_index

            if mod == 'COVERT':
                rows_with_one = (jobs_u_s[:, -2] == 1).nonzero()[0]
                z = np.zeros(len(rows_with_one))
                for i in range(len(rows_with_one)):
                    n = rows_with_one[i]
                    # job_id / arrival_time / processing_time / familiy_type / tardy_time / tardy_occur /mask / processed 여부
                    # remainning processing time, speed, current setup
                    st = setup[int(machine_matrix[chosen_index, 2])][int(jobs_u_s[n][3])]
                    pt = jobs_u_s[n][2] / machine_matrix[chosen_index, 1]
                    z[i] = -(1/pt*max(0,1-max(0,jobs_u_s[n][4]-pt-current_time)/pt))
                min_value = np.min(z)

                # 최소값의 인덱스들 찾기
                min_indices = np.where(z == min_value)[0]
                # 최소값 중 무작위로 하나 선택
                min_index = np.random.choice(min_indices)
                min_index = rows_with_one[min_index]
                action = min_index
            if mod == 'SST':
                rows_with_one = (jobs_u_s[:, -2] == 1).nonzero()[0]
                z = np.zeros(len(rows_with_one))
                for i in range(len(rows_with_one)):
                    n = rows_with_one[i]
                    # job_id / arrival_time / processing_time / familiy_type / tardy_time / tardy_occur /mask / processed 여부
                    # remainning processing time, speed, current setup
                    st = setup[int(machine_matrix[chosen_index, 2])][int(jobs_u_s[n][3])]
                    z[i] = st
                min_value = np.min(z)

                # 최소값의 인덱스들 찾기
                min_indices = np.where(z == min_value)[0]
                # 최소값 중 무작위로 하나 선택
                min_index = np.random.choice(min_indices)
                min_index = rows_with_one[min_index]
                action = min_index

            job_index=action.item()
            # job_id / arrival_time / processing_time / familiy_type / tardy_time / tardy_occur /mask / processed 여부
            machine_matrix[chosen_index,0]=jobs_u_s[job_index][2]/machine_matrix[chosen_index,1]+setup[int(machine_matrix[chosen_index,2])][int(jobs_u_s[job_index][3])]
            temp_setup=setup[int(machine_matrix[chosen_index,2])][int(jobs_u_s[job_index][3])]
            machine_matrix[chosen_index,2]=jobs_u_s[job_index][3]
            jobs_u_s[job_index,-2]=0
            jobs_u_s[job_index,-1]=1

            total_tardy+=max((current_time+machine_matrix[chosen_index,0])-jobs_u_s[job_index][-4],0)
            # job_id / arrival_time / processing_time / familiy_type / tardy_time / tardy_occur /mask / processed 여부
            jobs_u_s[job_index][-3]=max((current_time+machine_matrix[chosen_index,0])-jobs_u_s[job_index][-4],0)
            tardy=self.calculate_tardy(jobs_u_s,current_time)/100.0
            reward=past_tardy-tardy-temp_setup/100.0
            total_reward+=reward
            past_tardy=tardy
            
           
            #episode.append([job_state.clone(), machine_state.clone(), action, pi, reward, dones, mask, job_next_state.clone(), machine_next_state.clone()])
            if i==jobs.shape[0]-1:
                done=0
            else:
                done=1
            if mod=='RL':
                episode.append([state_tensor.clone(), action.item(), pi.item(), reward, done, mask.clone()])
            # job_id / arrival_time / processing_time / familiy_type / tardy_time / tardy_occur /mask / processed 여부
            machines.append("Machine "+str(chosen_index))
            jobss.append("Job "+str(job_index))
            start_times.append(current_time)
            durations.append(machine_matrix[chosen_index,0])
            
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

    #def evaluation(self,):


    def run_simulation(self,pr,sim_num,ppo,mod):
        episode=[]
        ave_tardy=[]
        for __ in range(pr):
            jobs,setup=self.generate_jobs()
            jobs[:,1:]=jobs[:,1:].astype(float)
            for _ in range(sim_num):
                machines,jobss,start_times,durations,episode,total_tardy,total_reward=self.schedule_jobs(jobs,setup,episode,ppo,mod)
                ave_tardy.append(total_tardy)
            #self.plot_gantt(machines,jobss,start_times,durations)
        ave_t=np.array(ave_tardy).mean()
        return ave_t,ave_tardy,episode
# 결과를 저장할 경우
    
# schedule_df.to_csv("schedule_results.csv", index=False) '''
# 결과를 저장할 경우
    
# schedule_df.to_csv("schedule_results.csv", index=False) '''
