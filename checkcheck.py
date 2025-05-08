import casadi as ca
import numpy as np
import spart_python.spart_casadi as ft
import spart_python.urdf2robot as u2r
from scipy.io import loadmat

def to_np(x):
    """SX/MX/DM 모두 → NumPy 배열"""
    return np.asarray(ca.DM(x))

# ------------------------------------------------------------------
# 1) 로봇 파라미터, MATLAB 검증 데이터 읽기
# ------------------------------------------------------------------
robot, _ = u2r.urdf2robot('SC_ur10e.urdf')
nq       = robot['n_q']
data     = loadmat('sim_values_only.mat')

# ------------------------------------------------------------------
# 2) 에러 로그 파일 초기화
# ------------------------------------------------------------------
with open('error_log.txt', 'w') as f:
    f.write("Error log started\n\n")

# ------------------------------------------------------------------
# 3) 샘플 루프
# ------------------------------------------------------------------
n_samples = data['R0'].shape[2]          # (= 2029)
for k in range(n_samples):

    # ---------- (1) 입력값 ----------
    R0_val = data['R0'][:,:,k]                 # (3×3)
    r0_val = data['r0'][k,:].reshape(3,1)      # (3×1)
    qm_val = data['qm'][k,:].reshape(nq,1)     # (n_q×1)
    u0_val = data['u0'][:,0,k].reshape(6,1)    # (6×1)
    um_val = data['um'][k,:].reshape(nq,1)     # (n_q×1)

    # ---------- (2) 순차 계산 ----------
    RJ, RL, rJ, rL, e, g = ft.kinematics(R0_val, r0_val, qm_val, robot)
    Bij, Bi0, P0, pm     = ft.diff_kinematics(R0_val, r0_val, rL, e, g, robot)
    t0, tL               = ft.velocities(Bij, Bi0, P0, pm, u0_val, um_val, robot)
    I0, Im               = ft.inertia_projection(R0_val, RL, robot)
    M0_tilde, Mm_tilde   = ft.mass_composite_body(I0, Im, Bij, Bi0, robot)
    H0, H0m, Hm          = ft.generalized_inertia_matrix(M0_tilde, Mm_tilde,
                                                         Bij, Bi0, P0, pm, robot)
    C0, C0m, Cm0, Cm     = ft.convective_inertia_matrix(t0, tL, I0, Im,
                                                         M0_tilde, Mm_tilde,
                                                         Bij, Bi0, P0, pm, robot)

    # ---------- (3) NumPy 변환 ----------
    C0   = to_np(C0);    C0m  = to_np(C0m)
    Cm0  = to_np(Cm0);   Cm   = to_np(Cm)
    H0   = to_np(H0);    H0m  = to_np(H0m);   Hm = to_np(Hm)
    P0   = to_np(P0);    pm   = to_np(pm)
    M0_tilde = to_np(M0_tilde) 
    

    # Bij, Bi0, Mm_tilde → 4‑D/3‑D 배열로 변환
    n_links = robot['n_links_joints']
    Bij_arr      = np.empty_like(data['Bij'][:,:,:,:,k])
    Bi0_arr      = np.empty_like(data['Bi0'][:,:,:,k])
    Mm_tilde_arr = np.empty_like(data['Mm_tilde'][:,:,:,k])

    for i in range(n_links):
        Bi0_arr[:,:,i]      = to_np(Bi0[i])
        Mm_tilde_arr[:,:,i] = to_np(Mm_tilde[i])
        for j in range(n_links):
            Bij_arr[:,:,i,j] = to_np(Bij[i][j])

    # ---------- (4) 오차 계산 ----------
    C0_err   = np.linalg.norm(C0   - data['C0'][:,:,k])
    C0m_err  = np.linalg.norm(C0m  - data['C0m'][:,:,k])
    Cm0_err  = np.linalg.norm(Cm0  - data['Cm0'][:,:,k])
    Cm_err   = np.linalg.norm(Cm   - data['Cm'][:,:,k])
    H0_err   = np.linalg.norm(H0   - data['H0'][:,:,k])
    H0m_err  = np.linalg.norm(H0m  - data['H0m'][:,:,k])
    Hm_err   = np.linalg.norm(Hm   - data['Hm'][:,:,k])
    P0_err   = np.linalg.norm(P0   - data['P0'][:,:,k])
    pm_err   = np.linalg.norm(pm   - data['pm'][:,:,k])
    Bij_err  = np.linalg.norm(Bij_arr  - data['Bij'][:,:,:,:,k])
    Bi0_err  = np.linalg.norm(Bi0_arr  - data['Bi0'][:,:,:,k])
    M0t_err  = np.linalg.norm(M0_tilde - data['M0_tilde'][:,:,k])
    Mmt_err  = np.linalg.norm(Mm_tilde_arr - data['Mm_tilde'][:,:,:,k])

    err_sum = (C0_err + C0m_err + Cm0_err + Cm_err + H0_err +
               H0m_err + Hm_err + P0_err + pm_err +
               Bij_err + Bi0_err + M0t_err + Mmt_err)

    # ---------- (5) 로그 ----------
    if err_sum > 1e-6:
        with open('error_log.txt', 'a') as f:
            f.write(f"Iteration {k}:\n")
            f.write(f"  C0_err   = {C0_err:.3e}\n")
            f.write(f"  C0m_err  = {C0m_err:.3e}\n")
            f.write(f"  Cm0_err  = {Cm0_err:.3e}\n")
            f.write(f"  Cm_err   = {Cm_err:.3e}\n")
            f.write(f"  H0_err   = {H0_err:.3e}\n")
            f.write(f"  H0m_err  = {H0m_err:.3e}\n")
            f.write(f"  Hm_err   = {Hm_err:.3e}\n")
            f.write(f"  P0_err   = {P0_err:.3e}\n")
            f.write(f"  pm_err   = {pm_err:.3e}\n")
            f.write(f"  Bij_err  = {Bij_err:.3e}\n")
            f.write(f"  Bi0_err  = {Bi0_err:.3e}\n")
            f.write(f"  M0~_err  = {M0t_err:.3e}\n")
            f.write(f"  Mm~_err  = {Mmt_err:.3e}\n\n")

# ------------------------------------------------------------------
# 4) 종료 메시지
# ------------------------------------------------------------------
with open('error_log.txt', 'a') as f:
    f.write("Error log ended\n")
