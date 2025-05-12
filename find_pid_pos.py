import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import copy

# 초기 PD 게인 설정
kp_init = np.array([7, 6, 3, 1, 0.1, 0.0001])
kd_init = np.sqrt(4 * kp_init)

# 저장할 데이터 변수 초기화
trajectory_data = {
    'time': [],
    'q_des': [],
    'q_actual': [],
    'tracking_error': []
}

# 시간 변화에 따른 목표 위치 생성 함수
def generate_target_trajectory(t, frequency=0.1):
    # Time-varying trajectory - sinusoidal pattern
    q1 = 0.5 * np.sin(2 * np.pi * frequency * t) + 0.5
    q2 = 0.3 * np.sin(2 * np.pi * frequency * t + np.pi/3) + 0.2
    q3 = 0.2 * np.sin(2 * np.pi * frequency * t + np.pi/4) + 0.3
    q4 = 0.1 * np.sin(2 * np.pi * frequency * t + np.pi/2)
    q5 = 0.05 * np.sin(2 * np.pi * frequency * t + 2*np.pi/3)
    q6 = 0.3 * np.sin(2 * np.pi * frequency * t + np.pi/6) + 0.3
    
    return np.array([q1, q2, q3, q4, q5, q6])

# PD 제어 콜백 함수
def ctrl_callback(model, data, desired_q, kp, kd):
    q_des = desired_q              # 목표 각도(라디언) 벡터
    qd_des = np.zeros(6)           # 목표 속도
    q = data.qpos[7:13]
    qd = data.qvel[6:12]
    tau = kp * (q_des - q) + kd * (qd_des - qd)
    data.ctrl[:6] = tau
    
    return q, q_des

# 주어진 PD 게인으로 시뮬레이션을 실행하고 성능 측정
def run_simulation(kp_factors, max_time=10.0, render=False):
    # kp_factors에서 실제 게인 계산
    kp = kp_init * kp_factors
    kd = np.sqrt(4 * kp)  # 임계 감쇠 계수 설정
    
    # 모델 로드
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # 데이터 저장 변수
    times = []
    errors = []
    q_actual_list = []
    q_desired_list = []
    
    # 시작 시간
    start_time = time.time()
    sim_time = 0.0
    
    # 뷰어 설정 (render=True인 경우)
    if render:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while sim_time < max_time:
                # 현재 시간에 해당하는 목표 위치 계산
                q_des = generate_target_trajectory(sim_time)
                
                # PD 제어 적용
                q_actual, q_des = ctrl_callback(model, data, q_des, kp, kd)
                
                # 시뮬레이션 스텝 진행
                mujoco.mj_step(model, data)
                
                # 추적 오차 계산 (RMSE)
                error = np.sqrt(np.mean((q_actual - q_des)**2))
                
                # 데이터 저장
                times.append(sim_time)
                errors.append(error)
                q_actual_list.append(q_actual.copy())
                q_desired_list.append(q_des.copy())
                
                # 뷰어 업데이트
                viewer.sync()
                
                # 시간 업데이트
                sim_time += model.opt.timestep
                
                # 실시간 속도 조절
                elapsed = time.time() - start_time
                if elapsed < sim_time:
                    time.sleep(sim_time - elapsed)
    else:
        # 렌더링 없이 시뮬레이션 실행 (최적화용)
        while sim_time < max_time:
            # 현재 시간에 해당하는 목표 위치 계산
            q_des = generate_target_trajectory(sim_time)
            
            # PD 제어 적용
            q_actual, q_des = ctrl_callback(model, data, q_des, kp, kd)
            
            # 시뮬레이션 스텝 진행
            mujoco.mj_step(model, data)
            
            # 추적 오차 계산 (RMSE)
            error = np.sqrt(np.mean((q_actual - q_des)**2))
            
            # 데이터 저장
            times.append(sim_time)
            errors.append(error)
            q_actual_list.append(q_actual.copy())
            q_desired_list.append(q_des.copy())
            
            # 시간 업데이트
            sim_time += model.opt.timestep
    
    # 결과 데이터 저장
    result = {
        'times': np.array(times),
        'errors': np.array(errors),
        'q_actual': np.array(q_actual_list),
        'q_desired': np.array(q_desired_list),
        'mean_error': np.mean(errors),  # 평균 오차 (목적 함수)
        'kp': kp,
        'kd': kd
    }
    
    return result

# 최적화 목적 함수
def objective_function(kp_factors):
    # 시뮬레이션 실행 (렌더링 없이)
    result = run_simulation(kp_factors, max_time=5.0, render=False)
    
    # 평균 오차 반환 (최소화할 목적 함수)
    return result['mean_error']

# 최적화 수행 함수
def optimize_pd_gains():
    # 초기 kp_factors는 1로 시작 (초기 kp 값 그대로 사용)
    initial_kp_factors = np.ones(6)
    
    # 최적화 경계 설정 (각 요소는 초기값의 0.1배에서 5배 사이)
    bounds = [(0.1, 5.0) for _ in range(6)]
    
    print("최적화 시작...")
    
    # 최적화 실행 (Powell 방법 사용 - 미분이 필요없는 방법)
    result = minimize(
        objective_function,
        initial_kp_factors,
        method='Powell',
        bounds=bounds,
        options={'maxiter': 20, 'disp': True}  # 최적화 반복 횟수 제한
    )
    
    # 최적화 결과 출력
    print("최적화 완료!")
    print(f"최적 kp_factors: {result.x}")
    
    # 최적 게인 계산
    optimal_kp = kp_init * result.x
    optimal_kd = np.sqrt(4 * optimal_kp)
    
    print(f"최적 kp: {optimal_kp}")
    print(f"최적 kd: {optimal_kd}")
    
    return optimal_kp, optimal_kd

# 결과 시각화 함수
def visualize_results(result):
    # 시간에 따른 추적 오차 그래프
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(result['times'], result['errors'])
    plt.title('Tracking Error Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('RMSE')
    plt.grid(True)
    
    # 각 관절 위치 추적 그래프
    plt.subplot(2, 1, 2)
    for i in range(6):
        plt.plot(result['times'], result['q_actual'][:, i], label=f'Joint {i+1} Actual')
        plt.plot(result['times'], result['q_desired'][:, i], '--', label=f'Joint {i+1} Desired')
    
    plt.title('Joint Position Tracking')
    plt.xlabel('Time (s)')
    plt.ylabel('Joint Position (rad)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('tracking_results.png')
    plt.show()

# 메인 함수
def main():
    global xml_path
    xml_path = "mujoco_src/spacerobot_cjt.xml"
    
    # 메뉴 선택
    print("작업 선택:")
    print("1. 최적의 PD 게인 찾기")
    print("2. 현재 PD 게인으로 시뮬레이션 실행")
    choice = input("선택 (1 또는 2): ")
    
    if choice == '1':
        # 최적 PD 게인 찾기
        optimal_kp, optimal_kd = optimize_pd_gains()
        
        # 최적 게인으로 시뮬레이션 실행 및 시각화
        print("최적 게인으로 시뮬레이션 실행...")
        result = run_simulation(optimal_kp / kp_init, max_time=10.0, render=True)
        
        # 결과 시각화
        visualize_results(result)
        
    elif choice == '2':
        # 현재 PD 게인으로 시뮬레이션 실행
        print("현재 게인으로 시뮬레이션 실행...")
        result = run_simulation(np.ones(6), max_time=15.0, render=True)
        
        # 결과 시각화
        visualize_results(result)
    
    else:
        print("잘못된 선택입니다. 1 또는 2를 입력하세요.")

if __name__ == "__main__":
    main()