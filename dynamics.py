import casadi as ca
import numpy as np
import spart_python.spart_casadi as ft  # Optimized spart_casadi module
import spart_python.urdf2robot as u2r

"""
Optimized casadi dynamics for free floating space robot 
x = [quat0, r0, qm, u0, um]
u0 : [6x1] Base-link velocities [ω, r_dot]. Angular velocity is expressed in a body-fixed CCS, while linear velocity is in the inertial CCS.
um : [nx1] Joint velocities, representing the motion of each joint in the system.
r0 : [3x1] Position of the base-link center-of-mass relative to the inertial frame.
quat0 : [4x1] Orientation of the base-link, represented as a quaternion.
qm : [nx1] Joint angle.
"""

file_name = 'SC_ur10e.urdf'
[robot, robot_key] = u2r.urdf2robot(file_name)

def dynamics(x, u):
    """
    Compute the system dynamics for a free-floating space robot
    
    Args:
        x: State vector [quat0, r0, qm, u0, um]
        u: Control input (joint torques)
        
    Returns:
        x_dot: Time derivative of the state vector
    """
    # Extract state components
    quat0 = x[0:4]
    r0 = x[4:7]
    qm = x[7:13]
    u0 = x[13:19]
    um = x[19:25]
    
    # Convert quaternion to DCM
    R0 = ft.quat_dcm(quat0)
    
    # Compute forward kinematics
    RJ, RL, rJ, rL, e, g = ft.kinematics(R0, r0, qm, robot)
    
    # Compute differential kinematics
    Bij, Bi0, P0, pm = ft.diff_kinematics(R0, r0, rL, e, g, robot)
    
    # Compute velocities
    t0, tL = ft.velocities(Bij, Bi0, P0, pm, u0, um, robot)
    
    # Compute inertia projections
    I0, Im = ft.inertia_projection(R0, RL, robot)
    
    N = ft.noc(r0, rL, P0, pm, robot)
    Ndot = ft.nocdot(r0, t0, rL, tL, P0, pm, robot)
    H = ft.generalized_inertia_matrix_noc(N, I0, Im, robot)
    C = ft.convective_inertia_matrix_noc(N, Ndot, t0, tL, I0, Im, robot)
    
    # Apply control input
    tau = ca.vertcat(ca.MX.zeros(6,1), u)
    
    # Combine velocities
    q_dot = ca.vertcat(u0, um)
    
    # Solve for accelerations
    q_ddot = ca.solve(H, (tau - ca.mtimes(C, q_dot)))
    
    # Compute quaternion derivative
    w = u0[0:3]
    quat_dot = ft.quat_dot(quat0, w)
    
    # Assemble state derivative
    x_dot = ca.vertcat(quat_dot, u0[3:6], um, q_ddot[0:12])
    
    return x_dot
    
def output(x):
    """
    Compute the output of the system (quaternion, position and twist of the end-effector)
    
    Args:
        x: State vector [quat0, r0, qm, u0, um]
        
    Returns:
        y: Output vector [quat_ee, r_ee, t_ee]
    """
    # Extract state components
    quat0 = x[0:4]
    r0 = x[4:7]
    qm = x[7:13]
    u0 = x[13:19]
    um = x[19:25]
    
    # Convert quaternion to DCM
    R0 = ft.quat_dcm(quat0)
    
    # Compute forward kinematics
    RJ, RL, rJ, rL, e, g = ft.kinematics(R0, r0, qm, robot)
    
    # Compute differential kinematics
    Bij, Bi0, P0, pm = ft.diff_kinematics(R0, r0, rL, e, g, robot)
    
    # Define end-effector transformation
    T_ee = ca.vertcat(
        ca.horzcat(ca.MX.eye(3), ca.vertcat(ca.MX(0), ca.MX(0.088), ca.MX(0))),
        ca.MX.zeros(1,4)
    )
    
    # Compute end-effector position
    r_ee = ca.mtimes(T_ee, ca.vertcat(rL[:,-1], 1))
    r_ee = r_ee[0:3]
    
    # Compute end-effector orientation
    quat_ee = ft.dcm_quat(ca.reshape(RL[:,-1], (3,3)))
    
    # Compute Jacobian at end-effector
    J0_ee, Jm_ee = ft.jacobian(r_ee, r0, rL, P0, pm, 6, robot)
    
    # Compute end-effector twist
    t_ee = ca.mtimes(J0_ee, u0) + ca.mtimes(Jm_ee, um)
    
    # Assemble output vector
    y = ca.vertcat(quat_ee, r_ee, t_ee)
    
    return y

def output_position(x):
    """
    Compute the position output of the system (position and twist of the end-effector)
    
    Args:
        x: State vector [quat0, r0, qm, u0, um]
        
    Returns:
        y: Output vector [r_ee, t_ee]
    """
    # Extract state components
    quat0 = x[0:4]
    r0 = x[4:7]
    qm = x[7:13]
    u0 = x[13:19]
    um = x[19:25]
    
    # Convert quaternion to DCM
    R0 = ft.quat_dcm(quat0)
    
    # Compute forward kinematics
    RJ, RL, rJ, rL, e, g = ft.kinematics(R0, r0, qm, robot)
    
    # Compute differential kinematics
    Bij, Bi0, P0, pm = ft.diff_kinematics(R0, r0, rL, e, g, robot)
    
    # Define end-effector transformation
    T_ee = ca.vertcat(
        ca.horzcat(ca.MX.eye(3), ca.vertcat(ca.MX(0), ca.MX(0.088), ca.MX(0))),
        ca.MX.zeros(1,4)
    )
    
    # Compute end-effector position
    r_ee = ca.mtimes(T_ee, ca.vertcat(rL[:,-1], 1))
    r_ee = r_ee[0:3]
    
    # Compute Jacobian at end-effector
    J0_ee, Jm_ee = ft.jacobian(r_ee, r0, rL, P0, pm, 6, robot)
    
    # Compute end-effector twist
    t_ee = ca.mtimes(J0_ee, u0) + ca.mtimes(Jm_ee, um)
    
    # Assemble output vector
    y = ca.vertcat(r_ee, t_ee)
    
    return y