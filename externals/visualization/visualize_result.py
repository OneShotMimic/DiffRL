# TODO: Visualize result using mujoco
import numpy as np
import mujoco_py
import os
import time
import scipy.spatial.transform as tf
from argparse import ArgumentParser

def load_states(exp_name):
    states = np.load(f"data/states/{exp_name}.npy")
    return states

def load_mujoco_model(env_name):
    model = mujoco_py.load_model_from_path(f"envs/assets/{env_name}.xml")
    sim = mujoco_py.MjSim(model)
    return model, sim

def set_sim_state(sim, qpos):
    old_state = sim.get_state()
    qvel_mj = np.zeros(len(qpos)-1)
    new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel_mj, old_state.act, old_state.udd_state)
    sim.set_state(new_state)
    sim.forward()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp_name", type=str, default=None, required=True)
    parser.add_argument("--env_name", type=str, default=None, required=True)
    args = parser.parse_args()

    states = load_states(args.exp_name)
    model, sim = load_mujoco_model(args.env_name)
    viewer = mujoco_py.MjViewer(sim)
    
    for i in range(1000):
        viewer.render()
        set_sim_state(sim, states[i])
        time.sleep(0.02)

