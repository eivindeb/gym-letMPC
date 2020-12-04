import pytest
import numpy as np
from gym_let_mpc.model import initialize_mpc_model
from gym_let_mpc.controllers import *


def test_robust_mpc(load_config_dict):
    config = load_config_dict
    config["mpc"]["uncertainty"] = 1
    with pytest.raises(NotImplementedError):
        initialize_mpc(initialize_mpc_model(config["plant"]["model"]), config["mpc"])
    del config["mpc"]["uncertainty"]

    config["mpc"]["params"]["n_robust"] = 1
    with pytest.raises(NotImplementedError):
        initialize_mpc(initialize_mpc_model(config["plant"]["model"]), config["mpc"])


def test_lqr_components(create_lqr_controller):
    lqr = create_lqr_controller
    assert lqr.K.shape == (1, 2) and np.allclose(lqr.K, np.array([1.3416, 1.0]), atol=1e-4)
    assert lqr.E.shape == (2,) and np.allclose(lqr.E, np.array([-0.2236, -0.1118]), atol=1e-5)
    assert lqr.S.shape == (2, 2) and np.allclose(lqr.S, np.array([[5.3666, 4.0], [4.0, 13.4164]]))
    prevK = np.copy(lqr.K)
    lqr.update_component(R=np.array([2]))
    assert not np.array_equal(prevK, lqr.K)


def test_mpc_objective(create_mpc):
    mpc = create_mpc
    assert str(mpc.mterm) == "sq((0.3-x1))"
    assert str(mpc.lterm) == "sq(x1)"


def test_cfs_update_reference(create_cfs):
    cfs = create_cfs
    assert cfs.mpc.mterm_fun([0.3], [0]).full()[0, 0] == (0.3-0.3)**2
    assert cfs.mpc.mterm_fun([0], [1]).full()[0, 0] == (0.3-0.0)**2
    assert cfs.mpc.lterm_fun([0], [1], [1], [1], [1]).full()[0, 0] == (0.0)**2

    new_ref = {"x1_r": 0.2}
    cfs.update_reference(new_ref)
    assert cfs.current_reference["x1_r"] == new_ref["x1_r"]
    assert cfs.mpc.mterm_fun([0.3], [0]).full()[0, 0] == (0.2-0.3)**2
    assert cfs.mpc.lterm_fun([0], [1], [1], [1], [1]).full()[0, 0] == (0.0)**2


def test_cfs_update_constraint(create_cfs):
    cfs = create_cfs

    cfs.update_constraints({"c-x1-u": 1.5})

# TODO: write test for MPC objective