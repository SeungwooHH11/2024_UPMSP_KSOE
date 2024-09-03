import torch, gc
import numpy as np

torch.backends.cudnn.benchmark = True
from Simulator_UPMSP import *
from MultiHeadAttention1 import *
import os
import vessl

gc.collect()
torch.cuda.empty_cache()
np.random.seed(1)
random.seed(1)
torch.manual_seed(1)

if __name__ == "__main__":
    problem_dir = '/output/problem_set/'
    if not os.path.exists(problem_dir):
        os.makedirs(problem_dir)
    model_dir = '/output/model/ppo/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    history_dir = '/output/history/'
    if not os.path.exists(history_dir):
        os.makedirs(history_dir)

    device = 'cuda'
    ppo = PPO(learning_rate=0.001, clipping_ratio=0.2, machine_len=12, d_model=512, num_heads=8, num_layers=2,
              dim_feedforward=1024).to(device)

    UPMSP = UPMSPScheduler(num_machines=12, initial_jobs=80, additional_jobs=10, additional_arrivals=5,
                           processing_time_range=(5, 15), tardy_time_range=(5, 15), start_additional_arrival=20,
                           arrival_interval=10, PPO=ppo)

    number_of_validation = 20
    number_of_validation_batch = 50
    number_of_problem = 3   # 한번에 몇개의 문제를
    number_of_batch = 5  # 문제당 몇 episode씩 한번에 학습할껀지
    number_of_trial = 1  # 1, 10, 100, 1000 #이를 몇번 반복할껀지
    number_of_iteration = int(1001 / number_of_trial)  # 전체 iteration #iteration 단위로 문제 변화
    validation = []
    validation_step = 10
    num_of_meta = 6
    history = np.zeros((number_of_iteration * number_of_trial, 2))
    validation_history = np.zeros((int(1001 / validation_step) + 10, number_of_validation))
    control = np.zeros((num_of_meta, number_of_validation))
    validation_set = []
    for j in range(number_of_validation):
        temp_jobs = UPMSP.generate_jobs()
        validation_set.append(temp_jobs.copy())

    '''
 mode_list = ['Random', 'SPT', 'SET', 'SRT', 'ATC', 'EDD', 'COVERT']
 temp_step = 0
 past_time_step=0

 for j in range(number_of_validation):
     B, T, b, tp, efi, nf, ef, dis, step_to_ij = Pr_sampler.sample()
     efi = efi.astype('int')
     validation.append([B, T, tp, b, efi, nf, ef, dis, step_to_ij, tardy_high])

     for nu, mod in enumerate(mode_list):
         rs = np.zeros(20)
         es = np.zeros(20)
         ts = np.zeros(20)
         for k in range(20):
             reward_sum, tardy_sum, ett_sum, event, episode, actions, probs, rewards, dones = simulation(
                 validation[j][0], validation[j][1], validation[j][2], validation[j][3], validation[j][4], validation[j][5],
                 validation[j][6], validation[j][7], validation[j][8], validation[j][9], mod, ppo)
             rs[k] = reward_sum
             es[k] = ett_sum
             ts[k] = tardy_sum
         Control_result[temp_step, nu, 0] = rs.mean()
         Control_result[temp_step, nu, 1] = rs.var()
         Control_result[temp_step, nu, 2] = es.mean()
         Control_result[temp_step, nu, 3] = es.var()
         Control_result[temp_step, nu, 4] = ts.mean()
         Control_result[temp_step, nu, 5] = ts.var()
     temp_step += 1
 for nu, mod in enumerate(mode_list):
     print(mod, Control_result[past_time_step:temp_step, nu, 0].mean(),
           Control_result[past_time_step:temp_step, nu, 2].mean(),
           Control_result[past_time_step:temp_step, nu, 4].mean())
 '''
    valid_step = 0
    k_epoch = 2
    for i in range(number_of_iteration):

        total_tardy, ave_tardy, episode = UPMSP.run_simulation(number_of_problem, number_of_batch)

        ave_loss, v_loss, p_loss = ppo.update(episode, k_epoch, number_of_problem * number_of_batch, i, model_dir)
        history[i, 0] = total_tardy
        vessl.log(step=i, payload={'train_average_reward': total_tardy})

        if i % validation_step == 0:
            valid_tardy = []
            best_tardy = []
            avve_tardy = 0
            for j in range(number_of_validation):
                temp_best_reward = 1000000
                ave_tardy = 0
                for k in range(number_of_validation_batch):
                    jobs = validation_set[j]
                    episode = []
                    machines, jobss, start_times, durations, episode, total_tardy, total_reward = UPMSP.schedule_jobs(
                        jobs, episode)
                    ave_tardy += (total_tardy)
                    temp_best_reward = min(temp_best_reward, total_tardy)
                validation_history[valid_step, j] = (ave_tardy / number_of_validation_batch)
                avve_tardy += (ave_tardy / number_of_validation_batch)
            print(ave_tardy / number_of_validation)
            valid_step += 1
            vessl.log(step=i, payload={'valid_average': avve_tardy / number_of_validation})

    history = pd.DataFrame(history)
    validation_history = pd.DataFrame(validation_history)
    history.to_excel(history_dir + 'history.xlsx', sheet_name='Sheet', index=False)
    validation_history.to_excel(history_dir + 'valid_history.xlsx', sheet_name='Sheet', index=False)
