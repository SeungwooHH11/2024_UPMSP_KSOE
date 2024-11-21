import torch,gc
gc.collect()
torch.cuda.empty_cache()
import numpy as np
torch.backends.cudnn.benchmark = True
from Simulator_PMSP_basic import *
from MultiHeadAttention_basic import *
import os
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
    PMSP=PMSPScheduler(num_machines=12, initial_jobs=80, additional_jobs=10, additional_arrivals=5, processing_time_range=(5, 15), machine_speed=[1,1,1,1,1,1,1.5,1.5,1.5,1.5,1.5,1.5], start_additional_arrival=20, arrival_interval=10,setup_range=(5,15),family_setup_num=8)
    number_of_validation=20
    number_of_validation_batch=50
    number_of_problem=4# 한번에 몇개의 문제를
    number_of_batch=12 # 문제당 몇 episode씩 한번에 학습할껀지
    number_of_trial=1  #1, 10, 100, 1000 #이를 몇번 반복할껀지
    number_of_iteration=int(3001/number_of_trial)  # 전체 iteration #iteration 단위로 문제 변화
    validation=[]
    validation_step = 100
    num_of_meta=6
    history = np.zeros((number_of_iteration * number_of_trial,2))
    validation_history=np.zeros((int(3001/validation_step)+10,number_of_validation))
    control=np.zeros((num_of_meta,number_of_validation))
    validation_set=[]
    for j in range(number_of_validation):
        temp_jobs=PMSP.generate_jobs()
        validation_set.append(temp_jobs.copy())

    name='TF1'
    filename = "Problem_set_" + +".xlsx"
    def save_arrays_to_excel(arrays, filename):
        # ExcelWriter 객체 생성
        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            for i, array in enumerate(arrays):
                # numpy 배열을 pandas DataFrame으로 변환
                df = pd.DataFrame(array)
                # 각 배열을 시트에 저장 (시트 이름은 Sheet1, Sheet2, ...)
                df.to_excel(writer, sheet_name=f"Sheet{i + 1}", index=False, header=False)
        print(f"{filename} 파일에 배열들이 시트별로 저장되었습니다.")


    # 함수 호출
    save_arrays_to_excel(validation_set,filename )