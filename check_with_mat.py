import spart_python.urdf2robot as u2r
import spart_python.spart_functions as ft
import numpy as np
from scipy.io import loadmat

file_name = 'SC_ur10e.urdf'
[robot, robot_key] = u2r.urdf2robot(file_name)
n = robot['n_q']

data = loadmat('sim_values_only.mat')
# print(data.keys())
C0_mat = data['C0']
C0m_mat = data['C0m']
Cm0_mat = data['Cm0']
Cm_mat = data['Cm']
H0_mat = data['H0']
H0m_mat = data['H0m']
Hm_mat = data['Hm']
P0_mat = data['P0']
pm_mat = data['pm']
Bij_mat = data['Bij']
Bi0_mat = data['Bi0']
M0_tilde_mat = data['M0_tilde']
Mm_tilde_mat = data['Mm_tilde']
# print('C0shpae',C0_mat.shape)
# print('C0mshape',C0m_mat.shape)
# print('Cm0shape',Cm0_mat.shape)
# print('Cmshape',Cm_mat.shape)
# print('H0shape',H0_mat.shape)
# print('H0mshape',H0m_mat.shape)
# print('Hmshape',Hm_mat.shape)
# print('P0shape',P0_mat.shape)
# print('pmshape',pm_mat.shape)
# print('Bijshape',Bij_mat.shape)
# print('Bi0shape',Bi0_mat.shape)
# print('M0_tildeshape',M0_tilde_mat.shape)
# print('Mm_tildeshape',Mm_tilde_mat.shape)


with open('error_log.txt', 'w') as f:
    f.write("Error log started\n\n")

for i in range(2029):
    r0 = data['r0'][i,:].reshape(3,1)
    R0 = data['R0'][:,:,i].reshape(3,3)
    qm = data['qm'][i,:].reshape(n,1)
    u0 = data['u0'][:,0,i].reshape(6,1)
    um = data['um'][i,:].reshape(n,1)
    RJ, RL, rJ, rL, e, g = ft.kinematics(R0, r0, qm, robot)
    Bij, Bi0, P0, pm = ft.diff_kinematics(R0, r0, rL, e, g, robot)
    t0, tL = ft.velocities(Bij, Bi0, P0, pm, u0, um, robot)
    r_cm = ft.center_of_mass(r0, rL, robot)
    I0, Im = ft.inertia_projection(R0, RL, robot)
    M0_tilde, Mm_tilde = ft.mass_composite_body(I0, Im, Bij, Bi0, robot)
    H0, H0m, Hm = ft.generalized_inertia_matrix(M0_tilde, Mm_tilde, Bij, Bi0, P0, pm, robot)
    C0, C0m, Cm0, Cm = ft.convective_inertia_matrix(t0, tL, I0, Im, M0_tilde, Mm_tilde, Bij, Bi0, P0, pm, robot)
    
    # save errors
    C0_error = np.linalg.norm(C0 - C0_mat[:,:,i])
    C0m_error = np.linalg.norm(C0m - C0m_mat[:,:,i])
    Cm0_error = np.linalg.norm(Cm0 - Cm0_mat[:,:,i])
    Cm_error = np.linalg.norm(Cm - Cm_mat[:,:,i])
    H0_error = np.linalg.norm(H0 - H0_mat[:,:,i])
    H0m_error = np.linalg.norm(H0m - H0m_mat[:,:,i])
    Hm_error = np.linalg.norm(Hm - Hm_mat[:,:,i])
    P0_error = np.linalg.norm(P0 - P0_mat[:,:,i])
    pm_error = np.linalg.norm(pm - pm_mat[:,:,i])
    Bij_error = np.linalg.norm(Bij - Bij_mat[:,:,:,:,i])
    Bi0_error = np.linalg.norm(Bi0 - Bi0_mat[:,:,:,i])
    M0_tilde_error = np.linalg.norm(M0_tilde - M0_tilde_mat[:,:,i])
    Mm_tilde_error = np.linalg.norm(Mm_tilde - Mm_tilde_mat[:,:,:,i])

    error_sum = C0_error + C0m_error + Cm0_error + Cm_error + H0_error + H0m_error + Hm_error + P0_error + pm_error + Bij_error + Bi0_error + M0_tilde_error + Mm_tilde_error
    # write errors to a file
    with open('error_log.txt', 'a') as f:
        if error_sum > 1e-6:
            f.write(f"Iteration {i}:\n")
            f.write(f"C0_error: {C0_error}\n")
            f.write(f"C0m_error: {C0m_error}\n")
            f.write(f"Cm0_error: {Cm0_error}\n")
            f.write(f"Cm_error: {Cm_error}\n")
            f.write(f"H0_error: {H0_error}\n")
            f.write(f"H0m_error: {H0m_error}\n")
            f.write(f"Hm_error: {Hm_error}\n")
            f.write(f"P0_error: {P0_error}\n")
            f.write(f"pm_error: {pm_error}\n")
            f.write(f"Bij_error: {Bij_error}\n")
            f.write(f"Bi0_error: {Bi0_error}\n")
            f.write(f"M0_tilde_error: {M0_tilde_error}\n")
            f.write(f"Mm_tilde_error: {Mm_tilde_error}\n")
# close the file
with open('error_log.txt', 'a') as f:
    f.write("All iterations completed.\n")
    f.write("Errors have been logged.\n")
