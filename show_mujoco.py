import mujoco
import mujoco.viewer
import numpy as np

# file path
xml_path = "spacerobot_cjt.xml"
# load mujoco model
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
data.qpos[7:13] = np.array([0.1, -0.3, -0.3, 0, 0, 0])
# set up mujoco viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    # run the simulation
    while True:
        mujoco.mj_step(model, data)
        # sync the viewer
        viewer.sync()
        