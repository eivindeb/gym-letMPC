import pytest
from gym_let_mpc.let_mpc import LetMPCEnv


def test_reset(create_cfs_env):
    env = create_cfs_env
    state = {"x1": 0.1}
    reference = {"x1_r": 0.2}
    constraint = {"c-x1-u": 1}
    env.reset(state, reference, constraint)
    assert env.control_system.history["state"][0]["x1"] == state["x1"]
    assert env.control_system.controller.current_reference["x1_r"] == reference["x1_r"]
    assert env.control_system.controller.constraints["c-x1-u"]["value"] == constraint["c-x1-u"]

    env.reset()

    assert env.control_system.current_state["x1"] != state["x1"]
    assert env.control_system.controller.current_reference["x1_r"] != reference["x1_r"]
    assert env.control_system.controller.constraints["c-x1-u"]["value"] != constraint["c-x1-u"]


#def test_sample_state(create_cfs_env)