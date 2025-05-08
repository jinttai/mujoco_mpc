import mujoco
import mujoco.viewer
import numpy as np
import casadi as ca
import spart_python.spart_functions as spart
import mpc_sol_position as mpc

# mujoco mpc test
xml_path = "spacerobot_cjt.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

pos_ref = np.array([0.5, 0.5, 0.5]).reshape(3,1)

y_ref = np.zeros((9,1))
#y_ref[0:4] = np.array([0, 0, 0, 1]).reshape(4,1)
y_ref[0:3] = pos_ref
y_ref[3:9] = np.array([0, 0, 0, 0, 0, 0]).reshape(6,1)
y_ref = ca.DM(y_ref)
mpc_ = mpc.MPCSolver()
iteration = 0


with mujoco.viewer.launch_passive(model,data) as viewer:
    while True:
        mujoco.mj_step(model, data)
        if iteration % 100 == 0:
            print("iteration: ", iteration)
            # mujoco state
            qpos = data.qpos
            qvel = data.qvel
            
            # mujoco state to spart state
            quat = np.append(qpos[4:7], qpos[3])
            R0 = spart.quat_dcm(quat)
            r0 = qpos[0:3]
            v0 = qvel[0:3]
            w0_i = qvel[3:6]
            w0_b = R0.T @ w0_i
            u0 = np.concatenate((w0_b, v0))
            qm = qpos[7:]
            um = qvel[6:]
            x_current = np.concatenate((quat, r0, qm, u0, um))
            x_current = x_current.reshape(25,1)
            # mpc solve
            u_mpc = mpc_.solve(x_current, y_ref)
            print("u_mpc: ", u_mpc[:, 0].full())
        site_id = model.site("end_effector").id
        #print(data.site_xpos[site_id])
        data.ctrl[6:12] = np.squeeze(u_mpc[:, 0].full())
        iteration += 1
        # mujoco viewer sync
        viewer.sync()