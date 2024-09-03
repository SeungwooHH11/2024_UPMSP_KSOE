import random
import pandas as pd

from MultiHeadAttention1 import *
import matplotlib.pyplot as plt
random.seed(42)
device = 'cuda'
import torch
np.random.seed(1)
random.seed(1)
torch.manual_seed(1)
class UPMSPScheduler:
    def __init__(self, num_machines=12, initial_jobs=80, additional_jobs=10, additional_arrivals=5,
                 processing_time_range=(5, 15), tardy_time_range=(5, 15), start_additional_arrival=20,
                 arrival_interval=10, PPO=None):
        self.num_machines = num_machines
        self.initial_jobs = initial_jobs
        self.additional_jobs = additional_jobs
        self.additional_arrivals = additional_arrivals
        self.processing_time_range = processing_time_range
        self.tardy_time_range = tardy_time_range
        self.start_additional_arrival = start_additional_arrival
        self.arrival_interval = arrival_interval
        self.MP = float(10 * (initial_jobs + additional_jobs * additional_arrivals) / 12)
        self.tardiness_factor = 0.1
        self.duedate_range = 0.5
        self.schedule = []
        self.ppo = PPO

    def generate_jobs(self):
        jobs = np.empty((0, 4 + self.num_machines))
        job_count = 0
        for i in range(self.initial_jobs):
            job_id = job_count
            arrival_time = 0
            property_j = [job_id, arrival_time]
            max_p = 0
            for j in range(self.num_machines):
                processing_time = random.randint(*self.processing_time_range)
                property_j.append(processing_time)
                max_p = max(max_p, processing_time)
            tardy_time = np.random.uniform(
                max(arrival_time + max_p, self.MP * (1 - self.tardiness_factor - self.duedate_range / 2)),
                self.MP * (1 - self.tardiness_factor + self.duedate_range / 2))

            property_j.append(tardy_time)
            property_j.append(1.0)
            jobs = np.vstack([jobs, np.array(property_j)])
            job_count += 1

        for j in range(self.additional_arrivals):
            arrival_time = self.start_additional_arrival + j * self.arrival_interval
            for k in range(self.additional_jobs):
                job_id = job_count
                property_j = [job_id, arrival_time]
                for j in range(self.num_machines):
                    processing_time = random.randint(*self.processing_time_range)
                    property_j.append(processing_time)
                tardy_time = np.random.uniform(
                    max(arrival_time + max_p, self.MP * (1 - self.tardiness_factor - self.duedate_range / 2)),
                    self.MP * (1 - self.tardiness_factor + self.duedate_range / 2))

                property_j.append(tardy_time)
                property_j.append(0.0)
                jobs = np.vstack([jobs, np.array(property_j)])
                # job_id / arrival time / pt0 pt0 pt0 pt0 pt0... pt0 pt0 / tardy / mask
                job_count += 1
        return jobs

    def get_agent(self, machine_end_time):
        # 리스트에서 최소값을 찾습니다.
        min_value = min(machine_end_time)
        # 최소값의 모든 인덱스를 찾습니다.
        min_indices = [i for i, x in enumerate(machine_end_time) if x == min_value]
        # 최소값의 인덱스 중 랜덤하게 하나를 선택합니다.
        chosen_index = random.choice(min_indices)
        return min_value, chosen_index

    def calculate_tardy(self, job_u_s, current_time):

        # job_id / arrival time / pt0 pt0 pt0 pt0 pt0... pt0 pt0 /tardy /mask/ processing 여부 / 발생 tardy
        tardy_occured = job_u_s[:, -1].sum()
        not_processed = job_u_s[job_u_s[:, -3] == 1].copy()

        min_process = not_processed[:, 2:-4].min(axis=1)
        tardy_yet = np.maximum(min_process + current_time - not_processed[:, -4], 0).sum()
        return tardy_occured + tardy_yet

    def schedule_jobs(self, jobs, episode):
        machines = []
        jobss = []
        start_times = []  # start times as floats
        durations = []  # durations as floats

        total_tardy = 0
        total_reward = 0

        past_tardy = 0
        # 각 머신의 작업 큐를 저장할 리스트 (우선순위 큐로 구현)
        machine_end_times = np.array([0] * self.num_machines)  # 각 머신의 작업 종료 시간 추적
        current_time = 0.0
        # job_id / arrival time / pt0 pt0 pt0 pt0 pt0... pt0 pt0 / processing 여부 / 발생 tardy
        job_len = jobs.shape[0]

        jobs_u_s = jobs.copy()
        new_column = np.zeros((job_len, 2))
        jobs_u_s = np.hstack((jobs_u_s, new_column))
        # job_id / arrival time / pt0 pt0 pt0 pt0 pt0... pt0 pt0 /tardy /mask/ processing 여부 / 발생 tardy

        for i in range(jobs.shape[0]):

            min_value, chosen_index = self.get_agent(machine_end_times)  # machine index
            current_time += min_value
            machine_end_times = machine_end_times - min_value
            # 조건: 1번째 열이 0이고, 2번째 열이 60보다 작은 행 필터링
            condition = (jobs_u_s[:, -2] == 0) & (jobs_u_s[:, 1] <= current_time)
            # 조건을 만족하는 행의 1번째 열 값을 1로 변경
            jobs_u_s[condition, -3] = 1

            count = np.sum(jobs_u_s[:, -3] == 1)
            while count == 0:
                filtered_arr = jobs_u_s[jobs_u_s[:, -2] == 0]
                min_row = np.min(filtered_arr[:, 1])
                gap = min_row - current_time
                current_time = min_row
                machine_end_times = machine_end_times - gap
                machine_end_times[machine_end_times < 0] = 0
                condition = (jobs_u_s[:, -2] == 0) & (jobs_u_s[:, 1] <= current_time)
                # 조건을 만족하는 행의 1번째 열 값을 1로 변경
                jobs_u_s[condition, -3] = 1
                count = np.sum(jobs_u_s[:, -3] == 1)

            mask = torch.tensor(jobs_u_s[:, -3].copy(), dtype=torch.float32).unsqueeze(0).to(device)

            job_state = jobs_u_s[:, 1:-3].copy()
            # job_id / arrival time / pt0 pt0 pt0 pt0 pt0... pt0 pt0 /tardy /mask/ processing 여부 / 발생 tardy
            job_state[:, 0] = job_state[:, 1 + chosen_index]
            job_state[:, -1] = job_state[:, -1] - current_time
            machine_state = machine_end_times.copy()
            job_state = torch.tensor(job_state, dtype=torch.float32).unsqueeze(0).to(device)
            machine_state = torch.tensor(machine_state, dtype=torch.float32).unsqueeze(0).to(device)

            action, pi = self.ppo.get_action(job_state, machine_state, mask)

            job_index = action.item()
            machine_end_times[chosen_index] = jobs_u_s[job_index][chosen_index + 2]
            jobs_u_s[job_index, -3] = 0
            jobs_u_s[job_index, -2] = 1
            total_tardy += max((current_time + jobs_u_s[job_index][chosen_index + 2]) - jobs_u_s[job_index][-4], 0)
            # job_id / arrival time / pt0 pt0 pt0 pt0 pt0... pt0 pt0 /tardy /mask/ processing 여부 / 발생 tardy
            jobs_u_s[job_index][-1] = max(
                (current_time + jobs_u_s[job_index][chosen_index + 2]) - jobs_u_s[job_index][-4], 0)
            tardy = self.calculate_tardy(jobs_u_s, current_time)
            reward = past_tardy - tardy
            total_reward += reward
            past_tardy = tardy

            job_next_state = jobs_u_s[:, 1:-3].copy()
            job_next_state[:, 0] = job_next_state[:, 1 + chosen_index]
            job_next_state[:, -1] = job_next_state[:, -1] - current_time
            machine_next_state = machine_end_times.copy()
            job_next_state = torch.tensor(job_next_state, dtype=torch.float32).unsqueeze(0).to(device)
            machine_next_state = torch.tensor(machine_next_state, dtype=torch.float32).unsqueeze(0).to(device)

            # episode.append([job_state.clone(), machine_state.clone(), action, pi, reward, dones, mask, job_next_state.clone(), machine_next_state.clone()])
            if i == jobs.shape[0] - 1:
                done = 0
            else:
                done = 1

            episode.append([job_state.clone(), machine_state.clone(), action.item(), pi.item(), reward, done, mask,
                            job_next_state.clone(), machine_next_state.clone()])

            machines.append("Machine " + str(chosen_index))
            jobss.append("Job " + str(job_index))
            start_times.append(current_time)
            durations.append(jobs_u_s[job_index][chosen_index + 2])

        return machines, jobss, start_times, durations, episode, total_tardy, total_reward

    def plot_gantt(self, machines, jobs, start_times, durations):
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

    def run_simulation(self, cat_num, cat_num2):
        episode = []
        ave_tardy = []
        for __ in range(cat_num):
            jobs = self.generate_jobs()
            jobs[:, 1:] = jobs[:, 1:].astype(float)
            for _ in range(cat_num2):
                machines, jobss, start_times, durations, episode, total_tardy, total_reward = self.schedule_jobs(jobs,
                                                                                                                 episode)
                ave_tardy.append(total_tardy)
            # self.plot_gantt(machines,jobss,start_times,durations)
        ave_t = np.array(ave_tardy).mean()
        return ave_t, ave_tardy, episode
# 결과를 저장할 경우

# schedule_df.to_csv("schedule_results.csv", index=False)
