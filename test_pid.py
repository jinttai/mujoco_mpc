import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# 전역 변수 선언
xml_path = "mujoco_src/spacerobot_cjt.xml"

# 최적 PD 게인 (제공된 값)
optimal_kp = np.array([100.593455, 45.00005263, 12.5932949, 3.09995538, 0.69789491, 0.003071458])
optimal_kd = np.array([1.00654459e-01, 1.07497233e-01, 1.91707960e-01, 5.01610700e-03, 8.47781584e-04, 1.47819615e-05])

# 사인파 궤적 생성 함수
def generate_sinusoidal_trajectory(t, frequency=0.1, phase_shift=True):
    """
    사인파 궤적 생성
    """
    qd1 = 0.3 * np.sin(2 * np.pi * frequency * t)
    qd2 = 0.2 * np.sin(2 * np.pi * frequency * t + (np.pi/3 if phase_shift else 0))
    qd3 = 0.15 * np.sin(2 * np.pi * frequency * t + (np.pi/4 if phase_shift else 0))
    qd4 = 0.1 * np.sin(2 * np.pi * frequency * t + (np.pi/2 if phase_shift else 0))
    qd5 = 0.05 * np.sin(2 * np.pi * frequency * t + (2*np.pi/3 if phase_shift else 0))
    qd6 = 0.2 * np.sin(2 * np.pi * frequency * t + (np.pi/6 if phase_shift else 0))
    
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

# 궤적 추적 시뮬레이션 실행
def run_trajectory_tracking_simulation(kp, kd, max_time=5.0, render=True):
    # 모델 로드
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # 중력을 0으로 설정
    model.opt.gravity = np.array([0, 0, 0])
    
    # 데이터 저장 변수
    times = []
    errors = []
    joint_errors = [[] for _ in range(6)]
    qd_actual_list = []
    qd_desired_list = []
    
    # 시작 시간
    start_time = time.time()
    sim_time = 0.0
    
    print("\n사인파 궤적 추적 시뮬레이션 시작...")
    
    # 뷰어 설정 (render=True인 경우)
    if render:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while sim_time < max_time:
                # 현재 시간에 해당하는 목표 속도 계산
                qd_des = generate_sinusoidal_trajectory(sim_time)
                
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
    else:
        # 렌더링 없이 시뮬레이션 실행
        while sim_time < max_time:
            # 현재 시간에 해당하는 목표 속도 계산
            qd_des = generate_sinusoidal_trajectory(sim_time)
            
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
    joint_mean_errors = [np.mean(joint_errors[i]) for i in range(6)]
    
    # 결과 출력
    print("\n사인파 궤적 추적 결과:")
    print(f"총 시뮬레이션 시간: {max_time:.2f}초")
    print(f"평균 추적 오차 (RMSE): {np.mean(errors):.6f}")
    print("각 관절별 평균 오차:")
    for i in range(6):
        print(f"관절 {i+1}: {joint_mean_errors[i]:.6f}")
    
    # 결과 데이터 저장
    result = {
        'times': np.array(times),
        'errors': np.array(errors),
        'joint_errors': joint_errors,
        'joint_mean_errors': joint_mean_errors,
        'qd_actual': np.array(qd_actual_list),
        'qd_desired': np.array(qd_desired_list),
        'mean_error': np.mean(errors),
        'kp': kp,
        'kd': kd
    }
    
    # 시뮬레이션 완료 후 오차 그래프 그리기
    plot_error_analysis(result)
    
    return result

# 향상된 오차 분석 그래프 함수
def plot_error_analysis(result):
    # 결과 디렉토리 생성
    os.makedirs("results", exist_ok=True)
    
    # 시각화 설정
    plt.style.use('ggplot')
    
    # Figure 1: 오차 시계열 및 통계 분석
    fig = plt.figure(figsize=(15, 10))
    
    # 1. 전체 RMSE 오차 그래프
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(result['times'], result['errors'], 'r-', linewidth=2)
    ax1.set_title('Overall Tracking Error (RMSE)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Error')
    ax1.grid(True, alpha=0.3)
    
    # 시간에 따른 오차의 통계 추가
    error_mean = np.mean(result['errors'])
    error_max = np.max(result['errors'])
    error_std = np.std(result['errors'])
    stats_text = f"Mean: {error_mean:.6f}\nMax: {error_max:.6f}\nStd: {error_std:.6f}"
    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. 각 관절별 오차 그래프
    ax2 = plt.subplot(2, 2, 2)
    cmap = plt.cm.get_cmap('tab10', 6)
    for i in range(6):
        ax2.plot(result['times'], result['joint_errors'][i], color=cmap(i), linewidth=1.5, label=f'Joint {i+1}')
    
    ax2.set_title('Individual Joint Tracking Errors', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Error')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # 3. 관절별 평균 오차 막대 그래프
    ax3 = plt.subplot(2, 2, 3)
    bars = ax3.bar(range(1, 7), result['joint_mean_errors'], color=cmap(range(6)))
    
    ax3.set_title('Average Error per Joint', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Joint')
    ax3.set_ylabel('Mean Error')
    ax3.set_xticks(range(1, 7))
    ax3.grid(True, axis='y', alpha=0.3)
    
    # 각 막대에 값 표시
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.5f}', ha='center', va='bottom', rotation=0, fontsize=8)
    
    # 4. 오차 분포 히스토그램
    ax4 = plt.subplot(2, 2, 4)
    ax4.hist(result['errors'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.axvline(error_mean, color='r', linestyle='--', linewidth=1.5, label=f'Mean: {error_mean:.5f}')
    
    ax4.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Error')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 전체 타이틀 설정
    plt.suptitle('Sinusoidal Trajectory Tracking Error Analysis', 
                fontsize=16, fontweight='bold')
    
    # 레이아웃 조정 및 저장
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('results/error_analysis.png', dpi=300, bbox_inches='tight')
    
    # Figure 2: 실제 vs 목표 속도 비교
    plt.figure(figsize=(15, 10))
    
    for i in range(6):
        plt.subplot(3, 2, i+1)
        plt.plot(result['times'], result['qd_actual'][:, i], 'b-', label='Actual')
        plt.plot(result['times'], result['qd_desired'][:, i], 'r--', label='Desired')
        plt.title(f'Joint {i+1} Velocity Tracking', fontsize=10, fontweight='bold')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (rad/s)')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/velocity_tracking.png', dpi=300, bbox_inches='tight')
    
    # Figure 3: 3D 궤적 시각화 (첫 3개 관절)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 실제 및 목표 궤적 (첫 3개 관절의 속도만 사용)
    ax.plot3D(result['qd_actual'][:, 0], result['qd_actual'][:, 1], result['qd_actual'][:, 2], 'red', label='Actual')
    ax.plot3D(result['qd_desired'][:, 0], result['qd_desired'][:, 1], result['qd_desired'][:, 2], 'blue', linestyle='--', label='Desired')
    
    ax.set_title('3D Velocity Trajectory (First 3 Joints)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Joint 1 Velocity (rad/s)')
    ax.set_ylabel('Joint 2 Velocity (rad/s)')
    ax.set_zlabel('Joint 3 Velocity (rad/s)')
    ax.legend()
    
    plt.savefig('results/3d_trajectory.png', dpi=300, bbox_inches='tight')
    
    # 그래프 표시
    plt.show()
    
    print("오차 분석 그래프가 'results' 디렉토리에 저장되었습니다.")

def main():
    print("최적 PD 게인을 이용한 사인파 궤적 추적 시뮬레이션")
    print(f"사용할 최적 kp: {optimal_kp}")
    print(f"사용할 최적 kd: {optimal_kd}")
    
    # 시뮬레이션 실행 시간 설정
    max_time = float( "20.0")
    
    # 시뮬레이션 실행
    run_trajectory_tracking_simulation(
        optimal_kp, optimal_kd, 
        max_time=max_time, 
        render=True
    )

if __name__ == "__main__":
    main()