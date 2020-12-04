import pytest
from gym_let_mpc.model import *


def test_supply_config_path(supply_config_path):
    with pytest.raises(NotImplementedError):
        initialize_mpc_model(supply_config_path)


def test_supply_config_dict(load_plant_config_dict):
    initialize_mpc_model(load_plant_config_dict["model"])


def test_non_linear_model(load_plant_config_dict):
    config_dict = load_plant_config_dict
    config_dict["model"]["class"] = "nonlinear"
    #with pytest.raises(NotImplementedError):
    #    initialize_mpc_model(config_dict)

    with pytest.raises(NotImplementedError):
        get_lqr_system(config_dict)


def test_mpc_model_correctness(create_mpc_model):
    model = create_mpc_model
    assert model.n_x == 2
    assert model.n_u == 1
    assert model.model_type == "continuous"
    assert mpc_model_get_variable_names(model, "_x") == ["x1", "x2"]
    assert mpc_model_get_variable_names(model, "_u") == ["u1"]
    # TODO: find more things to test here (i.e. process noise etc., correct evolution of states)


def test_lqr_model_correctness(load_config_dict):
    config_dict = load_config_dict
    A, B, Q, R, _JA, _JB = get_lqr_system({**config_dict["plant"]["model"], **config_dict["lqr"]})
    assert A.shape == (2, 2) and np.array_equal(A, np.array([[0, 0], [0.1, 0]]))
    assert B.shape == (2, 1) and np.array_equal(B, np.array([[0.25], [0]]))
    assert Q.shape == (2, 2) and np.array_equal(Q, np.array([[1, 0], [0, 1]]))
    assert R.shape == (1,) and np.array_equal(R, np.array([1]))
