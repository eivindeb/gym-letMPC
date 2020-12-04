import pytest
import json
from gym_let_mpc.model import *
from gym_let_mpc.controllers import *
from gym_let_mpc.let_mpc import LetMPCEnv


@pytest.fixture
def supply_config_path():
    return "test_config.json"


@pytest.fixture
def supply_env_config_path():
    return "test_config_env_2nd.json"


@pytest.fixture
def load_config_dict(supply_config_path):
    with open(supply_config_path) as file_object:
        config = json.load(file_object)

    if config["mpc"]["model"] == "plant":
        config["mpc"]["model"] = config["plant"]["model"]

    if config["lqr"]["model"] == "plant":
        config["lqr"]["model"] = config["plant"]["model"]
    elif config["lqr"]["model"] == "mpc":
        config["lqr"]["model"] = config["mpc"]["model"]

    return config


@pytest.fixture
def load_plant_config_dict(load_config_dict):
    return load_config_dict["plant"]


@pytest.fixture
def create_mpc_model(load_config_dict):
    return initialize_mpc_model(load_config_dict["mpc"]["model"])


@pytest.fixture
def create_lqr_model(load_config_dict):
    config_dict = load_config_dict
    return get_lqr_system(config_dict["lqr"])


@pytest.fixture
def create_mpc(create_mpc_model, load_config_dict):
    return initialize_mpc(create_mpc_model, load_config_dict["mpc"])


@pytest.fixture
def create_lqr_controller(create_lqr_model):
    A, B, Q, R, _JA, _JB = create_lqr_model
    return LQR(A, B, Q, R)


@pytest.fixture()
def create_cfs(load_config_dict):
    config_dict = load_config_dict
    return ETMPC(config_dict["mpc"], config_dict["lqr"])


@pytest.fixture
def create_cfs_env(supply_env_config_path):
    env = LetMPCEnv(supply_env_config_path)
    env.seed(0)
    return env
