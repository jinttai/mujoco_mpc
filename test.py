import mujoco as mj
import mujoco.viewer
import numpy as np
import time

file = "mujoco_src/spacerobot_cjt_fixed.xml"
model = mj.MjModel.from_xml_path(file)
data = mj.MjData(model)
sim_time = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    while True:
        if sim_time < 1:
            data.qvel[:12] = np.zeros(12)
        
        # 시뮬레이션 스텝 진행
        mj.mj_step(model, data)
        
        # 뷰어 업데이트
        viewer.sync()
        
        # 시간 업데이트
        time.sleep(model.opt.timestep)
        sim_time += model.opt.timestep