import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os

# 전역 변수 선언
xml_path = "mujoco_src/spacerobot_cjt.xml"

# 시간 변화에 따른 목표 속도 생성 함수
def generate_velocity_trajectory(t, frequency=0.1):
    # Time-varying velocity trajectory - sinusoidal pattern with different amplitudes for each joint
    qd1 = 0.3 * np.sin(2 * np.pi * frequency * t)
    qd2 = 0.2 * np.sin(2 * np.pi * frequency * t + np.pi/3)
    qd3 = 0.15 * np.sin(2 * np.pi * frequency * t + np.pi/4)
    qd4 = 0.1 * np.sin(2 * np.pi * frequency * t + np.pi/2)
    qd5 = 0.05 * np.sin(2 * np.pi * frequency * t + 2*np.pi/3)
    qd6 = 0.2 * np.sin(2 * np.pi * frequency * t + np.pi/6)
    
    return np.array([qd1, qd2, qd3, qd4, qd5, qd6])

# 속도 제어 콜백 함수
def vel_ctrl_callback(model, data, desired_qd, kp, kd):
    qd_des = desired_qd           # 목표 속도 벡터
    qdd_des = np.zeros(6)         # 목표 가속도 (일반적으로 0)
    qd = data.qvel[6:12]          # 현재 속도
    qdd = data.qacc[6:12]         # 현재 가속도
    
    # 속도 제어: 토크 = kp * (속도 오차) + kd * (가속도 오차)
    tau = kp * (qd_des - qd) + kd * (qdd_des - qdd)
    data.ctrl[:6] = tau
    
    return qd, qd_des

# 주어진 PD 게인으로 시뮬레이션을 실행하고 성능 측정 (속도 제어 버전)
def run_velocity_simulation(kp, kd, max_time=10.0, render=False):
    # 모델 로드
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # 데이터 저장 변수
    times = []
    errors = []
    joint_errors = [[] for _ in range(6)]  # 각 관절별 오차 저장
    qd_actual_list = []
    qd_desired_list = []
    
    # 시작 시간
    start_time = time.time()
    sim_time = 0.0
    
    # 뷰어 설정 (render=True인 경우)
    if render:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while sim_time < max_time:
                # 현재 시간에 해당하는 목표 속도 계산
                qd_des = generate_velocity_trajectory(sim_time)
                
                # 속도 제어 적용
                qd_actual, qd_des = vel_ctrl_callback(model, data, qd_des, kp, kd)
                
                # 시뮬레이션 스텝 진행
                mujoco.mj_step(model, data)
                
                # 추적 오차 계산 (RMSE)
                error = np.sqrt(np.mean((qd_actual - qd_des)**2))
                
                # 각 관절별 오차 계산
                for i in range(6):
                    joint_errors[i].append(abs(qd_actual[i] - qd_des[i]))
                
                # 데이터 저장
                times.append(sim_time)
                errors.append(error)
                qd_actual_list.append(qd_actual.copy())
                qd_desired_list.append(qd_des.copy())
                
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
            # 현재 시간에 해당하는 목표 속도 계산
            qd_des = generate_velocity_trajectory(sim_time)
            
            # 속도 제어 적용
            qd_actual, qd_des = vel_ctrl_callback(model, data, qd_des, kp, kd)
            
            # 시뮬레이션 스텝 진행
            mujoco.mj_step(model, data)
            
            # 추적 오차 계산 (RMSE)
            error = np.sqrt(np.mean((qd_actual - qd_des)**2))
            
            # 각 관절별 오차 계산
            for i in range(6):
                joint_errors[i].append(abs(qd_actual[i] - qd_des[i]))
            
            # 데이터 저장
            times.append(sim_time)
            errors.append(error)
            qd_actual_list.append(qd_actual.copy())
            qd_desired_list.append(qd_des.copy())
            
            # 시간 업데이트
            sim_time += model.opt.timestep
    
    # 각 관절별 평균 오차 계산
    joint_mean_errors = [np.mean(errors) for errors in joint_errors]
    
    # 결과 데이터 저장
    result = {
        'times': np.array(times),
        'errors': np.array(errors),
        'joint_errors': joint_errors,
        'joint_mean_errors': joint_mean_errors,
        'qd_actual': np.array(qd_actual_list),
        'qd_desired': np.array(qd_desired_list),
        'mean_error': np.mean(errors),  # 평균 오차 (목적 함수)
        'kp': kp,
        'kd': kd
    }
    
    return result

# 속도 제어 최적화 목적 함수
def velocity_objective_function(params):
    # 파라미터에서 kp, kd 값 추출
    # 첫 번째 6개 값은 kp, 다음 6개 값은 kd
    kp = params[:6]
    kd = params[6:]
    
    # 파라미터가 너무 작거나 너무 크면 페널티 부여
    if np.any(kp < 0.001) or np.any(kp > 100) or np.any(kd < 0.0001) or np.any(kd > 50):
        return 1000  # 큰 페널티 반환
    
    try:
        # 시뮬레이션 실행 (렌더링 없이)
        result = run_velocity_simulation(kp, kd, max_time=3.0, render=False)
        
        # 평균 오차 계산
        mean_error = result['mean_error']
        
        # 오버슈트와 불안정성에 대한 페널티 추가
        stability_penalty = 0
        for i in range(len(result['errors']) - 1):
            # 오차의 변화율이 큰 경우 (불안정성 지표)
            if i > 0:
                error_change = abs(result['errors'][i] - result['errors'][i-1])
                stability_penalty += error_change
        
        # 최종 목적 함수 = 평균 오차 + 안정성 페널티
        objective = mean_error + 0.01 * stability_penalty / len(result['errors'])
        
        return objective
    except Exception as e:
        # 오류가 발생하면 큰 페널티 반환
        print(f"Objective evaluation error: {str(e)}")
        return 1000

# 결과 시각화 함수
def visualize_velocity_results(result):
    # 결과 디렉토리 생성
    os.makedirs("results", exist_ok=True)
    
    # 시간에 따른 추적 오차 그래프
    plt.figure(figsize=(15, 15))
    
    plt.subplot(4, 1, 1)
    plt.plot(result['times'], result['errors'])
    plt.title('Velocity Tracking Error Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('RMSE')
    plt.grid(True)
    
    # 각 관절별 오차 그래프
    plt.subplot(4, 1, 2)
    for i in range(6):
        plt.plot(result['times'], result['joint_errors'][i], label=f'Joint {i+1} Error')
    plt.title('Individual Joint Errors')
    plt.xlabel('Time (s)')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    
    # 각 관절 속도 추적 그래프
    plt.subplot(4, 1, 3)
    for i in range(6):
        plt.plot(result['times'], result['qd_actual'][:, i], label=f'Joint {i+1} Actual Velocity')
    plt.title('Joint Velocity (Actual)')
    plt.xlabel('Time (s)')
    plt.ylabel('Joint Velocity (rad/s)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(4, 1, 4)
    for i in range(6):
        plt.plot(result['times'], result['qd_desired'][:, i], '--', label=f'Joint {i+1} Desired Velocity')
    plt.title('Joint Velocity (Desired)')
    plt.xlabel('Time (s)')
    plt.ylabel('Joint Velocity (rad/s)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/velocity_tracking_results.png')
    plt.show()

# 각 관절에 대해 개별적으로 초기 PD 게인 추정 함수
def estimate_joint_specific_pd_gains():
    print("각 관절별 초기 PD 게인 추정 중...")
    
    # 테스트할 게인 범위 (각 관절별로 다른 범위 설정)
    # 더 큰 관절(근위 관절)은 더 큰 게인 범위, 작은 관절(원위 관절)은 더 작은 게인 범위
    kp_ranges = [
        np.logspace(-1, 1, 3),    # 관절 1 (큰 관절)
        np.logspace(-1, 1, 3),    # 관절 2
        np.logspace(-1.5, 0.5, 3), # 관절 3
        np.logspace(-2, 0, 3),    # 관절 4
        np.logspace(-2.5, -0.5, 3), # 관절 5
        np.logspace(-3, -1, 3)     # 관절 6 (작은 관절)
    ]
    
    kd_ranges = [
        np.logspace(-2, 0, 3),    # 관절 1 (큰 관절)
        np.logspace(-2, 0, 3),    # 관절 2
        np.logspace(-2.5, -0.5, 3), # 관절 3
        np.logspace(-3, -1, 3),    # 관절 4
        np.logspace(-3.5, -1.5, 3), # 관절 5
        np.logspace(-4, -2, 3)     # 관절 6 (작은 관절)
    ]
    
    # 결과 저장 변수
    best_kp_per_joint = np.zeros(6)
    best_kd_per_joint = np.zeros(6)
    best_errors_per_joint = np.ones(6) * float('inf')
    
    # 각 관절별로 초기 게인 추정
    for joint_idx in range(6):
        print(f"\n관절 {joint_idx+1} 게인 추정 중...")
        
        for kp_val in kp_ranges[joint_idx]:
            for kd_val in kd_ranges[joint_idx]:
                # 테스트할 게인 생성
                # 현재 관절만 다른 값 사용, 나머지는 기본값으로 설정
                test_kp = np.ones(6) * 0.5  # 기본값
                test_kd = np.ones(6) * 0.05  # 기본값
                
                # 현재 테스트 중인 관절의 게인만 변경
                test_kp[joint_idx] = kp_val
                test_kd[joint_idx] = kd_val
                
                try:
                    # 짧은 시간 동안 시뮬레이션 실행
                    result = run_velocity_simulation(test_kp, test_kd, max_time=1.0, render=False)
                    
                    # 현재 관절의 오차만 확인
                    joint_error = np.mean(result['joint_errors'][joint_idx])
                    
                    print(f"관절 {joint_idx+1} 테스트 - kp: {kp_val:.4f}, kd: {kd_val:.4f}, error: {joint_error:.4f}")
                    
                    # 현재 관절에 대한 최적 게인 업데이트
                    if joint_error < best_errors_per_joint[joint_idx]:
                        best_errors_per_joint[joint_idx] = joint_error
                        best_kp_per_joint[joint_idx] = kp_val
                        best_kd_per_joint[joint_idx] = kd_val
                except Exception as e:
                    # 불안정한 게인은 건너뜀
                    print(f"관절 {joint_idx+1} 불안정한 게인 - kp: {kp_val:.4f}, kd: {kd_val:.4f}, 오류: {str(e)}")
                    continue
    
    # 모든 관절에 대한 최적 게인 출력
    print("\n각 관절별 추정된 초기 PD 게인:")
    for joint_idx in range(6):
        print(f"관절 {joint_idx+1}: kp = {best_kp_per_joint[joint_idx]:.4f}, kd = {best_kd_per_joint[joint_idx]:.4f}, error = {best_errors_per_joint[joint_idx]:.4f}")
    
    # 모든 관절에 대한 최적 게인 반환
    return best_kp_per_joint, best_kd_per_joint

# 추정된 게인을 기반으로 전체 관절에 대한 통합 시뮬레이션 수행
def evaluate_combined_gains(kp, kd):
    print("\n추정된 게인으로 통합 시뮬레이션 수행 중...")
    
    # 전체 관절 시뮬레이션 실행
    result = run_velocity_simulation(kp, kd, max_time=2.0, render=False)
    
    print(f"통합 게인 성능 - 평균 오차: {result['mean_error']:.6f}")
    print("각 관절별 평균 오차:")
    for joint_idx in range(6):
        print(f"관절 {joint_idx+1}: {np.mean(result['joint_errors'][joint_idx]):.6f}")
    
    return result

# 최적화를 통해 최적 PD 게인 찾기
def find_optimal_pd_gains(initial_kp, initial_kd):
    print("\n최적 PD 게인 찾기 시작...")
    
    # 초기 파라미터 설정 (kp와 kd를 하나의 배열로 합침)
    initial_params = np.concatenate([initial_kp, initial_kd])
    
    # 최적화 경계 설정 (초기 게인의 0.1배에서 10배 범위 내에서 탐색)
    bounds = []
    for i in range(6):
        # kp 경계
        kp_lower = max(0.001, initial_kp[i] * 0.1)
        kp_upper = min(100, initial_kp[i] * 10)
        bounds.append((kp_lower, kp_upper))
    
    for i in range(6):
        # kd 경계
        kd_lower = max(0.0001, initial_kd[i] * 0.1)
        kd_upper = min(50, initial_kd[i] * 10)
        bounds.append((kd_lower, kd_upper))
    
    # 최적화 실행
    result = minimize(
        velocity_objective_function,
        initial_params,
        method='Powell',  # Powell 방법 사용 (미분이 필요 없고, bound 제약에 유리)
        bounds=bounds,
        options={'maxiter': 100, 'disp': True}
    )
    
    # 최적화 결과에서 kp와 kd 추출
    optimal_kp = result.x[:6]
    optimal_kd = result.x[6:]
    
    print("최적화 완료!")
    print(f"초기 kp: {initial_kp}")
    print(f"초기 kd: {initial_kd}")
    print(f"최적 kp: {optimal_kp}")
    print(f"최적 kd: {optimal_kd}")
    
    return optimal_kp, optimal_kd

# 단계적인 최적화 수행
def hierarchical_optimization():
    # 1단계: 각 관절별 초기 게인 추정
    initial_kp, initial_kd = estimate_joint_specific_pd_gains()
    
    # 2단계: 추정된 초기 게인으로 통합 시뮬레이션 평가
    evaluate_combined_gains(initial_kp, initial_kd)
    
    # 3단계: 최적화를 통해 최적 게인 찾기
    optimal_kp, optimal_kd = find_optimal_pd_gains(initial_kp, initial_kd)
    
    # 4단계: 최적 게인 평가
    final_result = run_velocity_simulation(optimal_kp, optimal_kd, max_time=5.0, render=False)
    print(f"\n최종 최적 게인 성능 - 평균 오차: {final_result['mean_error']:.6f}")
    
    return optimal_kp, optimal_kd

def main():
    # 결과 디렉토리 생성
    os.makedirs("results", exist_ok=True)
    
    print("속도 제어를 위한 최적 PD 게인 찾기")
    
    # 단계적인 최적화 수행
    optimal_kp, optimal_kd = hierarchical_optimization()
    
    # 최적 게인으로 시뮬레이션 실행 및 결과 시각화
    print("\n최적 게인으로 시뮬레이션 실행 및 결과 시각화...")
    result = run_velocity_simulation(optimal_kp, optimal_kd, max_time=10.0, render=True)
    visualize_velocity_results(result)
    
    # 최적 게인 값 저장
    with open('results/optimal_velocity_pd_gains.txt', 'w') as f:
        f.write(f"Optimal kp: {optimal_kp}\n")
        f.write(f"Optimal kd: {optimal_kd}\n")
    
    print("\n최적 PD 게인을 'results/optimal_velocity_pd_gains.txt' 파일에 저장했습니다.")

if __name__ == "__main__":
    main()