import torch,gc
gc.collect()
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
from Simulator_PMSP_basic_Simplified import *
from MultiHeadAttention_Simplified import *
import os
import vessl
np.random.seed(1)
random.seed(1)
torch.manual_seed(1)

if __name__=="__main__":
    problem_dir='/output/problem_set/'
    if not os.path.exists(problem_dir):
        os.makedirs(problem_dir)
    model_dir='/output/model/ppo/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    history_dir='/output/history/'
    if not os.path.exists(history_dir):
        os.makedirs(history_dir)

    device='cuda'
    
    #PMSP=PMSPScheduler(12,48,48,3,(10,20),[1,1,1,1,1,1,1.5,1.5,1.5,1.5,1.5,1.5],1,80,80,(5,15),8)
    PMSP=PMSPScheduler(10,40,40,3,(10,20),[1,1,1,1,1,1.5,1.5,1.5,1.5,1.5],1,80,80,(5,15),6)
    
    #pr=PMSPScheduler(10,0,20,15,(10,20),[1,1,1,1,1,1.5,1.5,1.5,1.5,1.5],1,0,60,(5,15),10)
    #pr=PMSPScheduler(10,140,20,8,(10,20),[1,1,1,1,1,1.5,1.5,1.5,1.5,1.5],1,60,60,(5,15),10)
    ppo=PPO(learning_rate=0.001, clipping_ratio=0.2, machine_len=10, d_model=512, num_heads=8, fea_len=17,num_layers=1,dim_feedforward=1024).to(device)

    number_of_validation=20
    number_of_validation_batch=50
    number_of_problem=4# 한번에 몇개의 문제를
    number_of_batch=8 # 문제당 몇 episode씩 한번에 학습할껀지
    number_of_trial=1  #1, 10, 100, 1000 #이를 몇번 반복할껀지
    number_of_iteration=int(1001/number_of_trial)  # 전체 iteration #iteration 단위로 문제 변화
    validation=[]
    validation_step = 40
    num_of_meta=6
    history = np.zeros((number_of_iteration * number_of_trial,2))
    validation_history=np.zeros((int(1001/validation_step)+10,number_of_validation))
    control=np.zeros((num_of_meta,number_of_validation))
    validation_job=[]
    validation_setup=[]
    for j in range(number_of_validation):
        temp_jobs,setups=PMSP.generate_jobs()
        validation_job.append(temp_jobs.copy())
        validation_setup.append(setups.copy())
    
    

    mode_list = ['SSPT', 'SST', 'FIFO', 'ATCS', 'MDD', 'COVERT']
    for e,priority in enumerate(mode_list):
        avve_tardy=0
        for j in range(number_of_validation):
            temp_best_reward = 1000000
            ave_tardy=0
            for k in range(number_of_validation_batch):
                jobs=validation_job[j]
                setup=validation_setup[j]
                episode=[]
                machines,jobss,start_times,durations,episode,total_tardy,total_reward=PMSP.schedule_jobs(jobs,setup,episode,ppo,priority)
                ave_tardy+=(total_reward)
                temp_best_reward=min(temp_best_reward,total_tardy)
            control[e,j]=(ave_tardy/number_of_validation_batch)
            avve_tardy+=(ave_tardy/number_of_validation_batch)
        print(avve_tardy/number_of_validation)

    
    valid_step=0
    k_epoch=2
    for i in range(number_of_iteration):
        
        total_tardy,ave_tardy,episode=PMSP.run_simulation(number_of_problem,number_of_batch,ppo,'RL')
        ave_loss, v_loss, p_loss = ppo.update(episode, 160 ,k_epoch, i,model_dir)
        history[i,0]=total_tardy
        vessl.log(step=i, payload={'train_average_reward': total_tardy})
        vessl.log(step=i, payload={'ave_loss': ave_loss})
        vessl.log(step=i, payload={'v_loss': v_loss})
        vessl.log(step=i, payload={'p_loss': p_loss})
        

        
        if i%validation_step==0:
            valid_tardy=[]
            best_tardy=[]
            avve_tardy=0
            for j in range(number_of_validation):
                temp_best_reward = 1000000
                ave_tardy=0
                for k in range(number_of_validation_batch):
                    jobs=validation_job[j]
                    setup=validation_setup[j]
                    episode=[]
                    machines,jobss,start_times,durations,episode,total_tardy,total_reward=PMSP.schedule_jobs(jobs,setup,episode,ppo,'RL')
                    ave_tardy+=(total_reward)
                    temp_best_reward=min(temp_best_reward,total_tardy)
                validation_history[valid_step,j]=(ave_tardy/number_of_validation_batch)
                avve_tardy+=(ave_tardy/number_of_validation_batch)
            print(avve_tardy/number_of_validation)
            valid_step+=1
            
            vessl.log(step=i, payload={'valid_average':avve_tardy/number_of_validation})
            
                
    history=pd.DataFrame(history)
    validation_history=pd.DataFrame(validation_history)
    control=pd.DataFrame(control)
    control.to_excel(history_dir+'prrule.xlsx', sheet_name='Sheet', index=False)
    history.to_excel(history_dir+'history.xlsx', sheet_name='Sheet', index=False)
    validation_history.to_excel(history_dir + 'valid_history.xlsx', sheet_name='Sheet', index=False)
