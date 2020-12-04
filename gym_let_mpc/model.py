import do_mpc
import numpy as np
from gym_let_mpc.utils import str_replace_whole_words


def mpc_model_get_variable_names(model, var_type):  # TODO: find out why default control input state is inserted
    return [var_name for var_name in model[var_type].keys() if var_name != "default"]


def initialize_mpc_model(config):
    if isinstance(config, str):
        raise NotImplementedError
    elif not isinstance(config, dict):
        raise ValueError

    model = do_mpc.model.Model(config["type"])
    states = {}
    inputs = {}
    tvps = {}
    for state_name in sorted(config["states"]):
        states[state_name] = model.set_variable(var_type="_x", var_name=state_name, shape=(1, 1))

    for input_name in sorted(config["inputs"]):
        inputs[input_name] = model.set_variable(var_type="_u", var_name=input_name, shape=(1, 1))

    for tvp_name in sorted(config.get("tvps", {})):
        tvps[tvp_name] = model.set_variable(var_type="_tvp", var_name=tvp_name, shape=(1, 1))

    if config["class"] == "linear":
        for state_name, state_props in config["states"].items():
            model.set_rhs(state_name,
                          sum([a_s_val * states[a_s_name] for a_s_name, a_s_val in state_props["a"].items()])
                          + sum([b_u_val * inputs[b_u_name] for b_u_name, b_u_val in state_props["b"].items()]),
                          process_noise="W" in state_props)
    elif config["class"] == "nonlinear":
        state_names, input_names, tvp_names = list(states.keys()), list(inputs.keys()), list(tvps.keys())
        param_names = list(config.get("parameters", {}).keys())
        var_names = state_names + input_names + param_names + tvp_names
        for state_name, state_props in sorted(config["states"].items(), key=lambda x: len(x[0]), reverse=True):
            rhs_expr = state_props["rhs"]
            for var_name in var_names:
                if var_name in state_names:
                    rhs_expr = str_replace_whole_words(rhs_expr, var_name, 'states["{}"]'.format(var_name))
                elif var_name in input_names:
                    rhs_expr = str_replace_whole_words(rhs_expr, var_name, 'inputs["{}"]'.format(var_name))
                elif var_name in tvp_names:
                    rhs_expr = str_replace_whole_words(rhs_expr, var_name, 'tvps["{}"]'.format(var_name))
                else:
                    rhs_expr = str_replace_whole_words(rhs_expr, var_name, config["parameters"][var_name])
            model.set_rhs(state_name, eval(rhs_expr), process_noise="W" in state_props)
    else:
        raise NotImplementedError

    model.setup()

    return model


def get_lqr_system(config):
    def expression_replace_parameters(expr, parameters):
        for p_name, p_val in sorted(parameters.items(), key=lambda x: len(x[0]), reverse=True):
            expr = str_replace_whole_words(expr, p_name, p_val)

        return expr

    if isinstance(config, str):
        raise NotImplementedError
    elif not isinstance(config, dict):
        raise ValueError

    A, B = [], []
    if config["model"]["class"] not in ["linear", "linearization"]:
        raise NotImplementedError

    state_order = sorted(config["model"]["states"].keys())
    input_order = sorted(config["model"]["inputs"].keys())

    for state_name in state_order:
        s_a, s_b = [], []
        for a_s_name in state_order:
            if a_s_name in config["model"]["states"][state_name].get("a", {}):
                a_s_val = config["model"]["states"][state_name]["a"][a_s_name]
                if isinstance(a_s_val, str):
                    a_s_val = expression_replace_parameters(a_s_val, config["model"]["parameters"])
                    if config["model"]["class"] == "linear":
                        a_s_val = eval(a_s_val)
            else:
                a_s_val = 0
            s_a.append(a_s_val)
        A.append(s_a)

        for b_u_name in input_order:
            if b_u_name in config["model"]["states"][state_name].get("b", {}):
                b_u_val = config["model"]["states"][state_name]["b"][b_u_name]
                if isinstance(b_u_val, str):
                    b_u_val = expression_replace_parameters(b_u_val, config["model"]["parameters"])
                    if config["model"]["class"] == "linear":
                        b_u_val = eval(b_u_val)
            else:
                b_u_val = 0
            s_b.append(b_u_val)
        B.append(s_b)

    Q = np.array(config["objective"]["Q"])
    R = np.array(config["objective"]["R"])

    if config["model"]["class"] == "linear":
        A = np.array(A)
        B = np.array(B)
        JA, JB = None, None
    elif config["model"]["class"] == "linearization":
        JA, JB = A, B
        A, B = None, None

    return A, B, Q, R, JA, JB




