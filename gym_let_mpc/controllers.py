import do_mpc
import scipy.linalg
from gym_let_mpc.model import *
import matplotlib.pyplot as plt
import matplotlib.lines
import collections.abc
from gym_let_mpc.utils import str_replace_whole_words, casadiNNVF, TensorFlowEvaluator
import casadi
import copy
import time


def initialize_mpc(model, config, tvp_fun=None, p_fun=None, suppress_IPOPT_output=True, linear_solver="MA27", value_function=None):
    if config["params"].get("n_robust", 0) > 0:
        raise NotImplementedError  # TODO: if using robust MPC

    mpc = do_mpc.controller.MPC(model)
    mpc.set_param(**config["params"])
    nlpsol_opts = config.get("nlpsol_opts", {})#{"ipopt.linear_solver": linear_solver}
    if suppress_IPOPT_output:
        nlpsol_opts.update({"ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": 0})
    else:
        nlpsol_opts["ipopt.print_user_options"] = "yes"
    #nlpsol_opts["ipopt.warm_start_init_point"] = "yes"

    mpc.set_param(nlpsol_opts=nlpsol_opts)

    reference = config.get("reference", None)
    mpc = mpc_set_objective(mpc, config["objective"], reference=reference, parameters=config["model"].get("parameters", None), value_function=value_function)

    if tvp_fun is not None:
        mpc.set_tvp_fun(lambda t: mpc.get_tvp_template())
        mpc.tvp_fun = tvp_fun
    if p_fun is not None:
        p_template = mpc.get_p_template(1)
        mpc.set_p_fun(lambda t: p_template)
        mpc.p_fun = p_fun
    if "constraints" in config:
        mpc = mpc_set_constraints(mpc, config["constraints"])
    if "terminal_constraints" in config:
        mpc = mpc_set_constraints(mpc, config["terminal_constraints"], terminal=True)
    if "scaling" in config:
        mpc = mpc_set_scaling(mpc, config["scaling"])
    if "uncertainty" in config:
        raise NotImplementedError  # TODO: if using robust MPC

    mpc.setup()

    return mpc


def mpc_set_objective(mpc, cost_parameters, reference=None, parameters=None, value_function=None):
    costs = {"lterm": None, "mterm": None}
    for cost_term in costs.keys():
        if cost_term in cost_parameters:
            expr = cost_parameters[cost_term]["expression"]
            if expr == "0":
                costs[cost_term] = casadi.DM(0)
            else:
                for var in sorted(cost_parameters[cost_term]["variables"], key=lambda x: len(x["name"]), reverse=True):
                    if var["type"] in ["_x", "_u", "_tvp", "_p", "_aux"]:
                        expr = str_replace_whole_words(expr, var["name"], 'mpc.model.{}["{}"]'.format(var["type"][1:], var["name"]))
                    elif var["type"] == "_r":
                        assert reference is not None and var["name"] in reference
                        expr = str_replace_whole_words(expr, var["name"], reference[var["name"]]["value"])
                    elif var["type"] == "parameter":
                        assert parameters is not None and var["name"] in parameters
                        expr = str_replace_whole_words(expr, var["name"], parameters[var["name"]])
                    else:
                        raise ValueError

                costs[cost_term] = eval(expr)

    if cost_parameters.get("vf", None) is not None:
        vf = casadiNNVF(**cost_parameters["vf"])
    else:
        vf = None

    mpc.set_objective(mterm=costs["mterm"], lterm=costs["lterm"], vf=vf, discount_factor=cost_parameters.get("discount_factor", 1))

    if "R_delta" in cost_parameters:
        mpc.set_rterm(**cost_parameters["R_delta"])

    return mpc


def mpc_set_constraints(mpc, constraints, terminal=False):
    for c in constraints:
        if "variables" in c:
            assert not terminal
            expr = c["expression"]
            for var in sorted(c["variables"], key=lambda x: len(x["name"]), reverse=True):
                if var["type"] in ["_x", "_u", "_tvp", "_aux"]:
                    expr = str_replace_whole_words(expr, var["name"],
                                                   'mpc.model.{}["{}"]'.format(var["type"][1:], var["name"]))
                else:
                    raise ValueError
            expr_name = c["name"]
            eps_names = [s["expr_name"] for s in mpc.nl_cons_list]
            while expr_name in eps_names:
                expr_name += "1"
            mpc.set_nl_cons(expr_name, eval(expr), ub=c["value"], soft_constraint=c.get("soft", False), penalty_term_cons=c["cost"])
        else:
            if c.get("soft", False):  # TODO: support for arbitrary expressions
                assert not terminal
                if c["type"] == "_x":
                    expr = mpc.model.x[c["name"]]
                elif c["type"] == "_u":
                    expr = mpc.model.u[c["name"]]
                else:
                    raise NotImplementedError

                if c["constraint_type"] == "lower":
                    expr = expr * -1
                    c["value"] *= -1

                mpc.set_nl_cons("csoft-{}-{}".format(c["name"], c["constraint_type"][0]),
                                expr, soft_constraint=True, penalty_term_cons=c["cost"],
                                ub=c["value"])
            else:
                if terminal:
                    mpc.use_terminal_bounds = True
                    if isinstance(c["value"], str):
                        val = 0
                    else:
                        val = c["value"]
                    mpc.terminal_bounds[c["constraint_type"], c["name"]] = val
                else:
                    mpc.bounds[c["constraint_type"], c["type"], c["name"]] = c["value"]

    return mpc


def mpc_set_scaling(mpc, scaling):
    for s in scaling:
        mpc.scaling[s["type"], s["state_name"]] = s["value"]

    return mpc


def mpc_get_solution(mpc, states=None, inputs=None, t_ind=-1):  # TODO: need to improve this
    """Get MPC predicted states x^hat and input sequence of length control horizon computed for timestep t_ind."""
    assert states is not None or inputs is not None

    if states is not None:
        if states == "all":
            states = mpc_model_get_variable_names(mpc.model, "_x")
        state_preds = []
        for state in states:
            state_preds.append(mpc.data.prediction(("_x", state), t_ind))
        state_preds = np.concatenate(state_preds, axis=0)
        if inputs is None:
            return state_preds
    if inputs is not None:
        if inputs == "all":
            inputs = mpc_model_get_variable_names(mpc.model, "_u")
        input_sequences = []
        for control_input in inputs:
            input_sequences.append(mpc.data.prediction(("_u", control_input), t_ind))
        input_sequences = np.concatenate(input_sequences, axis=0)
        if states is None:
            return input_sequences

    return state_preds, input_sequences


def mpc_get_aux_value(mpc, aux_name, h_idx=0):
    aux_idx = mpc.model["_aux"].labels().index("[{},0]".format(aux_name))
    return mpc.opt_aux_num["_aux", h_idx, 0][aux_idx].__float__()


def mpc_get_algstate_value(mpc, algstate_name, h_idx=0):
    z_idx = mpc.model["_z"].labels().index("[{},0]".format(algstate_name))
    return mpc.opt_x_num_unscaled["_z", h_idx, 0, 0][z_idx].__float__()


class LQR:
    def __init__(self, A, B, Q, R, JA=None, JB=None):
        self.A, self.B, self.Q, self.R, self.JA, self.JB = A, B, Q, R, JA, JB
        self.K, self.S, self.E = None, None, None
        if self.A is not None and self.B is not None:
            self._compute_control_law()

    def get_action(self, x):
        return - self.K * x

    def _compute_control_law(self):  # TODO: supress pending deprecated warnings about matrix
        """Solve the continuous time lqr controller.

        dx/dt = A x + B u

        cost = integral x.T*Q*x + u.T*R*u
        """

        # ref Bertsekas, p.151

        # first, try to solve the ricatti equation
        self.S = np.matrix(scipy.linalg.solve_discrete_are(self.A, self.B, self.Q, self.R))

        # compute the LQR gain
        #self.K = np.matrix(scipy.linalg.inv(np.atleast_2d(self.R)) * (self.B.T * self.S))
        self.K = np.matrix(scipy.linalg.inv(np.atleast_2d(self.B.T * self.S * self.B + self.R)) * self.B.T * self.S * self.A)

        self.E, eigVecs = scipy.linalg.eig(self.A - self.B * self.K)

    def update_component(self, **kwargs):
        assert len(kwargs) > 0
        self.__dict__.update(**kwargs)
        self._compute_control_law()

    def evaluate_linear_model(self, operating_point):
        expr_JA, expr_JB = str(self.JA).replace("\'", ""), str(self.JB).replace("\'", "")
        for var_name, var_val in sorted(operating_point.items(), key=lambda x: len(x[0]), reverse=True):
            if isinstance(var_val, np.ndarray):
                var_val = var_val[0]
            expr_JA = str_replace_whole_words(expr_JA, var_name, var_val)
            expr_JB = str_replace_whole_words(expr_JB, var_name, var_val)

        self.A, self.B = np.array(eval(expr_JA)), np.array(eval(expr_JB))
        self._compute_control_law()


class LMPC:
    def __init__(self, mpc_config, mpc_model=None, viewer=None):
        self.mpc_config = mpc_config

        if mpc_model is None:
            mpc_model = initialize_mpc_model(mpc_config["model"])

        self._tvp_data = None
        if len(mpc_config["model"].get("tvps", [])) > 0:
            tvp_fun = self._get_tvp_values
            self.tvp_props = mpc_config["model"]["tvps"]
        else:
            tvp_fun = None
            self.tvp_props = None

        for t_c in mpc_config.get("terminal_constraints", []):
            if isinstance(t_c["value"], str):
                assert t_c["value"] in self.tvp_props

        if len(mpc_config["model"].get("ps", [])) > 0:
            p_fun = self._get_p_values
        else:
            p_fun = None

        self.value_function = None
        self.viewer = viewer

        self.input_names = sorted(mpc_config["model"]["inputs"].keys())
        self.state_names = sorted(mpc_config["model"]["states"].keys())
        self.reference_names = sorted(mpc_config.get("reference", {}).keys())

        self.current_input = {u_name: 0 for u_name in self.input_names}
        self.current_reference = {r_name: 0 for r_name in
                                  self.reference_names}

        self._initialize_mpc(mpc_model=mpc_model)

        if tvp_fun is not None:
            self._tvp_template = self.mpc.get_tvp_template()
        if p_fun is not None:
            self._p_template = self.mpc.get_p_template(1)

        self.constraints = {}
        for c in mpc_config["constraints"]:
            if "expression" in c:
                c_type = "cnlin"
                bound_type = None
            else:
                if c.get("soft"):
                    c_type = "clins"
                else:
                    c_type = "clin"
                bound_type = c.get("constraint_type", "u")[0]
            self.constraints["{}-{}{}".format(c_type, c["name"], "-{}".format(bound_type) if bound_type is not None else "")] = c

        self.initial_state = np.zeros((self.mpc.model.n_x,))

        self.mpc_state_preds = None
        self._mpc_action_sequence = None

        self.history = None
        self.extra_render_axes = 0

    def reset(self, state=None, reference=None, constraint=None, tvp=None):
        if reference is not None:
            self.update_reference(reference)

        if constraint is not None:
            self.update_constraints(constraint)

        self.mpc.reset_history()
        if self.viewer is not None:
            self.viewer["mpc_graphics"].clear()

        if state is None:
            state = self.initial_state
        elif isinstance(state, dict):  # TODO: partial state intialization
            state = self._get_state_vector(state)
        self.mpc.x0 = state
        self.mpc.u0 = np.zeros((len(self.input_names),))

        if tvp is not None:
            assert self.tvp_props is not None
            self._tvp_data = {name: val for name, val in tvp.items()}
        elif self.tvp_props is not None:
            self._tvp_data = {name: None for name in self.tvp_props}

        self.mpc.set_initial_guess()
        #self.mpc.set_initial_p_num()
        #self.mpc.calculate_aux_num()

        self.current_input = {u_name: None for u_name in self.input_names}
        self.history = {"inputs": [copy.deepcopy(self.current_input)], "references": [copy.deepcopy(self.current_reference)],
                        "errors": [self._get_tracking_error(state)],
                        "tvp": []}

    def get_action(self, state, *args, tvp_values=None, **kwargs):
        raise NotImplementedError

    def configure_viewer(self, viewer=None, plot_prediction=True):
        if viewer is None:  # TODO: fix checking of render property in config of state and input here
            fig, axes = plt.subplots(self.mpc.model.n_x + self.mpc.model.n_u, sharex=False, figsize=(9, 16))
            axes = {k: axes[i] for i, k in enumerate(mpc_model_get_variable_names(self.mpc.model, "_x") +
                                                     mpc_model_get_variable_names(self.mpc.model, "_u"))}
            viewer = {
                "fig": fig,
                "axes": axes
            }
        self.viewer = viewer
        self.viewer["mpc_plot_prediction"] = plot_prediction
        self.viewer["mpc_graphics"] = do_mpc.graphics.Graphics(self.mpc.data)
        self.viewer["constraint_lines"] = {}
        self.viewer["reference_lines"] = {}
        if plot_prediction:
            for state in mpc_model_get_variable_names(self.mpc.model, "_x"):
                self.viewer["mpc_graphics"].pred_lines["_x", state] = \
                    self.viewer["axes"][state].plot([], [], label="{}_pred".format(state), marker="x", markersize=15,
                                                    linestyle="dashed")

            for control_input in mpc_model_get_variable_names(self.mpc.model, "_u"):
                self.viewer["mpc_graphics"].pred_lines["_u", control_input] = \
                    self.viewer["axes"][control_input].plot([], [], label="{}_pred".format(control_input), marker="x",
                                                            markersize=15, drawstyle="steps", linestyle="dashed")

        for r_name in self.reference_names:
            state_name = r_name[:-2]
            self.viewer["reference_lines"][r_name] = self.viewer["axes"][state_name].add_line(
                matplotlib.lines.Line2D([], [], label=r_name,
                                        color=next(self.viewer["axes"][state_name]._get_lines.prop_cycler)["color"])
            )

        for constraint_name, c_props in self.constraints.items():
            if not constraint_name.startswith("cnlin-") and c_props.get("render", True):
                self._add_constraint_plot_line(constraint_name)

        return self.viewer

    def _add_constraint_plot_line(self, constraint_name):
        _, state_name, c_type = constraint_name.split("-")
        other_constraint_name = constraint_name[:-1] + ("u" if c_type == "l" else "l")
        if other_constraint_name in self.viewer["constraint_lines"]:
            color = self.viewer["constraint_lines"][other_constraint_name].get_color()
        else:
            color = next(self.viewer["axes"][state_name]._get_lines.prop_cycler)["color"]
        self.viewer["constraint_lines"][constraint_name] = self.viewer["axes"][state_name].add_line(
            matplotlib.lines.Line2D([], [], label="c-{}".format(constraint_name[-1]),
                                    color=color, linestyle="dotted" if c_type == "u" else "dashdot")
        )

    def render(self, show=True):  # TODO: Assert store_full_solution is true for prediction plotting
        if self.viewer is None:
            self.configure_viewer()

        if self.viewer["mpc_plot_prediction"]:
            mpc_compute = self.history["mpc_compute"] if "mpc_compute" in self.history else [True] * len(self.history["tvp"])
            pred_line_data = [{"x": [], "y": []} for i in range(len(self.viewer['mpc_graphics'].pred_lines.full))]

            mpc_computation_inds = [t_ind for t_ind, computed in enumerate(mpc_compute) if computed]

            for pred_i, mpc_computation_t_ind in enumerate(mpc_computation_inds):
                self.viewer["mpc_graphics"].plot_predictions(t_ind=pred_i)
                for line_i, line in enumerate(self.viewer['mpc_graphics'].pred_lines.full):
                    line_x, line_y = line.get_data()
                    if len(pred_line_data[line_i]["x"]) == 0 or min(line_x) >= max(pred_line_data[line_i]["x"]):
                        pred_line_data[line_i]["x"].extend(line_x)
                        pred_line_data[line_i]["y"].extend(line_y)
                    else:
                        overlap_data = [abs(x_d - min(line_x)) for x_d in pred_line_data[line_i]["x"]]
                        overlap_point = min(range(len(overlap_data)),
                                            key=overlap_data.__getitem__)  # Due to float rounding, the exact number might not be present in the other list
                        pred_line_data[line_i]["x"] = pred_line_data[line_i]["x"][:overlap_point] + list(line_x)
                        pred_line_data[line_i]["y"] = pred_line_data[line_i]["y"][:overlap_point] + list(line_y)

            if "tvp_lines" in self.viewer:
                tvp_line_data = {tvp_name: [] for tvp_name in self.tvp_props}
                for comp_i in range(len(mpc_computation_inds)):
                    if comp_i == len(mpc_computation_inds) - 1:  # Last iteration
                        forecast_i_length = self.mpc.n_horizon
                    else:
                        forecast_i_length = mpc_computation_inds[comp_i + 1] - mpc_computation_inds[comp_i]
                    for tvp_name, tvp_data in tvp_line_data.items():
                        tvp_data.extend(self.history["tvp"][mpc_computation_inds[comp_i]][tvp_name][:forecast_i_length])
                for tvp_name, tvp_lines in self.viewer["tvp_lines"].items():
                    tvp_data_x = np.arange(len(tvp_line_data[tvp_name])) * self.mpc.data.meta_data['t_step']
                    tvp_lines["forecast"].set_data(tvp_data_x, tvp_line_data[tvp_name])
                    mpc_computation_markers = mpc_compute + [False] * (
                            tvp_data_x.shape[0] - len(mpc_compute))
                    tvp_lines["forecast"].set_markevery(mpc_computation_markers)

            for line_i, (data, line) in enumerate(zip(pred_line_data, self.viewer['mpc_graphics'].pred_lines.full)):
                mpc_computation_markers = mpc_compute + [False] * (
                        len(pred_line_data[line_i]["x"]) - len(mpc_compute))
                line.set_data(data["x"], data["y"])
                line.set_markevery(mpc_computation_markers)

        ref_data_y = {r_name: [step_ref[r_name] for step_ref in self.history["references"]]
                      for r_name in self.history["references"][0]}
        ref_data_x = np.array(list(range(len(self.history["references"])))) * self.mpc.data.meta_data['t_step']
        for r_name, r_data in ref_data_y.items():
            self.viewer["reference_lines"][r_name].set_data(ref_data_x, r_data)

        for c_name, c_props in self.constraints.items():
            if c_name in self.viewer["constraint_lines"]:
                if c_props.get("soft", False) and c_props["constraint_type"] == "lower":
                    c_val = c_props["value"] * -1
                else:
                    c_val = c_props["value"]
                self.viewer["constraint_lines"][c_name].set_data(
                    [0, (len(self.history["inputs"]) - 1) * self.mpc.data.meta_data['t_step']],
                    [c_val, c_val]
                )

        if show:
            self.viewer["fig"].show()

    def _get_state_vector(self, state):
        """
        Get controller state vector (x) from state dictionary.
        :param state: (dict) dictionary of string state name keys with float state values.
        :returns (np.ndarray of (float)) state vector with order as defined in self.state_names
        """
        return np.array([state[s_name] for s_name in self.state_names]).reshape(-1, 1)

    def _get_state_dict(self, state_vector):
        return {s_name: state_vector[s_i] for s_i, s_name in enumerate(self.state_names)}

    def _get_input_dict(self, input_vector):
        return {u_name: input_vector[u_i] for u_i, u_name in enumerate(self.input_names)}

    def _get_tracking_error(self, state):
        if isinstance(state, np.ndarray):
            state = self._get_state_dict(state)
        error = {}
        for r_name in self.current_reference:
            s_name = r_name[:-2]
            error[s_name] = state[s_name] - self.current_reference[r_name]
        return error

    def _get_tvp_values(self, t_now):
        tvp_order = sorted(self._tvp_data)
        for t in range(self.mpc.n_horizon + 1):
            self._tvp_template["_tvp", t] = np.array([self._tvp_data[k][t] for k in tvp_order])

        return self._tvp_template

    def _get_p_values(self, t_now):
        raise NotImplementedError

    def update_reference(self, reference):  # TODO: changing mpc components require setting up new model, thereby throwing away data. Maybe not allow during operation. Change reference to be TVP
        if len(reference) > 0:
            assert reference.keys() <= self.current_reference.keys()
            self.current_reference.update(reference)
            for ref_name, ref_props in self.mpc_config.get("reference", {}).items():
                ref_props["value"] = self.current_reference[ref_name]
            self._initialize_mpc(mpc_model=self.mpc.model)

    def update_constraints(self, constraints):  # TODO: fix for TVPmodels
        if len(constraints) > 0:
            constraints = {k: {"value": v} for k, v in constraints.items()}
            for c_name in constraints:
                c_name_parts = c_name.split("-")
                if c_name not in self.constraints:
                    self.constraints[c_name] = {"name": c_name_parts[1],
                                                "type": "_{}".format(c_name_parts[1][0]),
                                                "constraint_type": "upper" if c_name_parts[-1] == "u" else "lower"}
                    self._add_constraint_plot_line(c_name)
                    self.mpc_config["constraints"].append(self.constraints[c_name])
                self.constraints[c_name]["value"] = constraints[c_name]["value"]

                for c_i, c in enumerate(self.mpc_config["constraints"]):
                    if c["name"] == c_name_parts[1] and c["constraint_type"] == (
                    "upper" if c_name_parts[-1] == "u" else "lower"):
                        self.mpc_config["constraints"][c_i]["value"] = constraints[c_name]["value"]
            self._initialize_mpc(mpc_model=self.mpc.model)

    def update_model(self, model):
        def update_dict_recursively(d, u):
            for k, v in u.items():
                if isinstance(v, collections.abc.Mapping):
                    d[k] = update_dict_recursively(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        assert model["states"].keys() <= self.mpc_config["model"]["states"].keys()
        self.mpc_config["model"] = update_dict_recursively(self.mpc_config["model"], model)
        self._initialize_mpc()

    def update_mpc_params(self, params, state=None):
        self.mpc_config["params"].update(params)
        self._initialize_mpc(state=state)

    def _update_mpc_data(self, old_data, new_data):
        data_fields = ["_x", "_u", "_y", "_z", "_tvp", "_p", "success"]
        for data_field in data_fields:
            setattr(old_data, data_field, getattr(new_data, data_field))

        return old_data

    def _initialize_mpc(self, mpc_model=None, state=None):
        if hasattr(self, "mpc"):
            old_data = self.mpc.data
        else:
            old_data = None
        if mpc_model is None:
            mpc_model = initialize_mpc_model(self.mpc_config["model"])
        self.mpc = initialize_mpc(mpc_model,
                                  self.mpc_config,
                                  tvp_fun=self._get_tvp_values if self.tvp_props is not None else None,
                                  p_fun=self._get_p_values if len(self.mpc_config["model"].get("ps", [])) > 0 else None,
                                  value_function=self.value_function)
        if old_data is not None:
            self.mpc.data = self._update_mpc_data(self.mpc.data, old_data)
        self.mpc.x0 = state if state is not None else np.zeros_like(self.mpc.x0.master)
        self.mpc.u0 = np.array(
            [self.current_input[i_name] if self.current_input[i_name] is not None else 0 for i_name in
             self.input_names]).reshape(-1, 1)
        self.mpc.set_initial_guess()
        if self.viewer is not None and "mpc_graphics" in self.viewer:
            self.viewer["mpc_graphics"].data = self.mpc.data

    def set_value_function(self, input_phs, output_ph, tf_session):
        assert (isinstance(input_phs, list) or isinstance(input_phs, tuple)) and len(input_phs) == 2
        self.value_function = TensorFlowEvaluator(input_phs, [output_ph], tf_session)
        self._initialize_mpc(mpc_model=self.mpc.model)

    def set_value_function_weights_and_biases(self, weights, biases):
        assert isinstance(self.mpc.vf, casadiNNVF)
        self.mpc.vf.set_weights_and_biases(weights, biases)

    def save_value_function(self, save_folder, name=""):
        assert isinstance(self.mpc.vf, casadiNNVF)
        self.mpc.vf.save_to_file(save_folder, name)

    def load_value_function(self, load_folder, name=""):
        assert isinstance(self.mpc.vf, casadiNNVF)
        self.mpc.vf.load_from_file(load_folder, name)


class ETMPC(LMPC):
    def __init__(self, mpc_config, lqr_config, mpc_model=None, viewer=None):
        assert mpc_config.get("params", {}).get("store_full_solution", False)
        super().__init__(mpc_config, mpc_model=mpc_model, viewer=viewer)
        self.lqr_config = lqr_config

        A, B, Q, R, JA, JB = get_lqr_system(lqr_config)
        self.lqr = LQR(A, B, Q, R, JA, JB)

        self.steps_since_mpc_computation = None

    def reset(self, state=None, reference=None, constraint=None, tvp=None):
        super().reset(state=state, reference=reference, constraint=constraint, tvp=tvp)
        self.steps_since_mpc_computation = None
        self.mpc_state_preds = None
        self._mpc_action_sequence = None

        if self.lqr_config["model"]["class"] == "linear":
            self.mpc.u0 = self.lqr.get_action(state)

        self.mpc.set_initial_guess()

        self.history.update({"epsilons": [], "state_preds": [], "u_mpc": [], "u_lqr": [], "u_cfs": [], "mpc_compute": [], "execution_time": []})

    def get_action(self, state, compute_mpc_solution=False, tvp_values=None):  # TODO: This is maybe bad as the name doesnt suggest that it changes the object but it acts closely related to a step function
        if self.steps_since_mpc_computation is None or self.steps_since_mpc_computation >= self.mpc.n_horizon - 1:
            compute_mpc_solution = True

        for ref_name in self.current_reference:
            if ref_name in tvp_values:
                self.current_reference[ref_name] = tvp_values[ref_name][0]

        state_vec = self._get_state_vector(state)
        self._tvp_data = tvp_values


        if compute_mpc_solution:
            for t_c in self.mpc_config.get("terminal_constraints", []):
                if isinstance(t_c["value"], str):
                    self.mpc.terminal_bounds[t_c["constraint_type"], t_c["name"]] = tvp_values[t_c["value"]][-1]

            self.mpc.t0 = len(self.history["mpc_compute"]) * self.mpc.data.meta_data['t_step']
            mpc_optimal_action = self.mpc.make_step(state_vec)
            self.mpc_state_preds = mpc_get_solution(self.mpc, states="all")
            self._mpc_action_sequence = mpc_get_solution(self.mpc, inputs="all")

            u_cfs = mpc_optimal_action
            self.steps_since_mpc_computation = 0

            self.history["mpc_compute"].append(True)
            epsilon_dict = self._get_state_dict(np.full_like(self.mpc_state_preds[:, 0, :], np.nan))
            if self.tvp_props is not None:
                epsilon_dict.update({name: 0 for name in self.tvp_props})
            self.history["epsilons"].append(epsilon_dict)
            self.history["state_preds"].append(np.full_like(self.mpc_state_preds[:, 0, :], np.nan))
            self.history["u_mpc"].append(mpc_optimal_action)
            self.history["u_lqr"].append(np.full_like(mpc_optimal_action, np.nan))
            self.history["execution_time"].append(sum([v for k, v in self.mpc.solver_stats.items() if k.startswith("t_proc")]))
        else:
            mpc_state_pred = self.mpc_state_preds[:, self.steps_since_mpc_computation + 1]
            if self.lqr_config["model"]["class"] == "linearization" and self.steps_since_mpc_computation == 0:
                operating_point = self._get_state_dict(state_vec)
                operating_point.update(self._get_input_dict(self._mpc_action_sequence[0, 1, :]))
                self.lqr.evaluate_linear_model(operating_point)

            # TODO: get preds and stuff from model?
            epsilon = state_vec - mpc_state_pred
            exec_time = time.process_time()
            u_lqr = np.array(self.lqr.get_action(epsilon))
            exec_time = time.process_time() - exec_time
            u_mpc = self._mpc_action_sequence[:, self.steps_since_mpc_computation + 1]
            u_cfs = u_mpc + u_lqr

            self.history["mpc_compute"].append(False)
            epsilon_dict = self._get_state_dict(epsilon)
            if self.tvp_props is not None:
                epsilon_dict.update({name: self._tvp_data[name][0] -
                                           self.history["tvp"][-(self.steps_since_mpc_computation + 1)][name][self.steps_since_mpc_computation + 1]
                                     for name in self._tvp_data})
            self.history["epsilons"].append(epsilon_dict)
            self.history["state_preds"].append(mpc_state_pred)
            self.history["u_mpc"].append(u_mpc)
            self.history["u_lqr"].append(u_lqr)
            self.history["execution_time"].append(exec_time)

            self.steps_since_mpc_computation += 1

        for u_i, u_name in enumerate(self.input_names):
            if "clin-{}-u".format(u_name) in self.constraints:
                u_cfs[u_i] = min(u_cfs[u_i], self.constraints["clin-{}-u".format(u_name)]["value"])
            if "clin-{}-l".format(u_name) in self.constraints:
                u_cfs[u_i] = max(u_cfs[u_i], self.constraints["clin-{}-l".format(u_name)]["value"])

        self.history["u_cfs"].append(u_cfs)

        for input_i, input_name in enumerate(self.input_names):
            self.current_input[input_name] = u_cfs[input_i]

        self.history["inputs"].append(copy.deepcopy(self.current_input))
        self.history["references"].append(copy.deepcopy(self.current_reference))
        self.history["errors"].append(self._get_tracking_error(state))
        self.history["tvp"].append(self._tvp_data)

        return u_cfs

    def configure_viewer(self, viewer=None):
        super().configure_viewer(viewer=viewer)
        self.viewer["lqr_lines"] = {}

        for control_input in mpc_model_get_variable_names(self.mpc.model, "_u"):
            self.viewer["lqr_lines"][control_input] = self.viewer["axes"][control_input].add_line(
                matplotlib.lines.Line2D([], [], label="{}_lqr".format(control_input),
                                        color=next(self.viewer["axes"][control_input]._get_lines.prop_cycler)["color"])
            )

        return self.viewer

    def render(self, show=True):
        super().render(show=False)

        lqr_data = np.array(self.history["u_lqr"])
        lqr_data_x = np.array(list(range(len(self.history["u_lqr"])))) * self.mpc.data.meta_data['t_step']
        for u_i, control_input in enumerate(mpc_model_get_variable_names(self.mpc.model, "_u")):
            self.viewer["lqr_lines"][control_input].set_data(lqr_data_x, lqr_data[:, u_i])

        if show:
            self.viewer["fig"].show()

    def update_model(self, model):
        def update_dict_recursively(d, u):
            for k, v in u.items():
                if isinstance(v, collections.abc.Mapping):
                    d[k] = update_dict_recursively(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        super().update_model(model=model)
        self.lqr_config["model"] = update_dict_recursively(self.lqr_config["model"], model)
        A, B, Q, R = get_lqr_system(self.lqr_config)
        self.lqr.update_component(A=A, B=B)

    # TODO: add support for changing Q and R in MPC and LQR


class AHMPC(LMPC):
    def __init__(self, mpc_config, mpc_model=None, viewer=None):  # TODO: maybe ensure u_t>hend = 0 by constraints or cost
        if "nlpsol_opts" not in mpc_config:
            mpc_config["nlpsol_opts"] = {}

        mpc_config["nlpsol_opts"]["ipopt.max_iter"] = 250
        for state in mpc_config["model"]["states"].values():
            state["rhs"] = "({}) * (1 - hend)".format(state["rhs"])
        
        mpc_config["objective"]["lterm"]["expression"] = "({}) * (1 - hend)".format(mpc_config["objective"]["lterm"]["expression"])
        mpc_config["objective"]["lterm"]["variables"].append({"name": "hend", "type": "_tvp"})
        for input_name in mpc_config["model"]["inputs"]:
            if input_name not in [var["name"] for var in mpc_config["objective"]["lterm"]["variables"]]:
                mpc_config["objective"]["lterm"]["variables"].append({"name": input_name, "type": "_u"})
            mpc_config["objective"]["lterm"]["expression"] += " + hend * ({}) ** 2".format(input_name)

        for constraint in mpc_config["constraints"]:
            if constraint.get("soft", False):
                if "expression" in constraint:
                    constraint["expression"] = "({}) * (1 - hend)".format(constraint["expression"])
                    constraint["variables"].append({"name": "hend", "type": "_tvp"})
                else:
                    constraint["variables"] = [{"name": "hend", "type": "_tvp"},
                                                {"name": constraint["name"], "type": constraint["type"]}]
                    constraint["expression"] = "({}) * (1 - hend)".format(constraint["name"])
                    if constraint.get("constraint_type", "upper") == "lower":
                        constraint["value"] *= -1
                        constraint["expression"] = "-" + constraint["expression"]

        if "tvps" not in mpc_config["model"]:
            mpc_config["model"]["tvps"] = {}
        mpc_config["model"]["tvps"]["hend"] = {
            "true": [{
                "type": "constant",
                "kw": {
                    "value": 0
                }
            }]
        }
        if "ps" not in mpc_config["model"]:
            mpc_config["model"]["ps"] = {}
        mpc_config["model"]["ps"]["n_horizon"] = {}

        super().__init__(mpc_config, mpc_model=mpc_model, viewer=viewer)

    def reset(self, state=None, reference=None, constraint=None, tvp=None):
        super().reset(state=state, reference=reference, constraint=constraint, tvp=tvp)

        self.mpc.set_initial_guess()

        self.history["mpc_horizon"] = []

    def get_action(self, state, n_horizon, tvp_values=None):  # TODO: This is maybe bad as the name doesnt suggest that it changes the object but it acts closely related to a step function
        state_vec = self._get_state_vector(state)
        if n_horizon < self.mpc_config["params"]["n_horizon"]:
            tvp_values["hend"][n_horizon:] = [1.0] * (self.mpc_config["params"]["n_horizon"] + 1 - n_horizon)

        for ref_name in self.current_reference:
            self.current_reference[ref_name] = tvp_values[ref_name][0]

        self.history["mpc_horizon"].append(n_horizon)

        self._tvp_data = tvp_values

        mpc_optimal_action = self.mpc.make_step(state_vec)
        self.mpc_state_preds = mpc_get_solution(self.mpc, states="all")
        self._mpc_action_sequence = mpc_get_solution(self.mpc, inputs="all")

        for input_i, input_name in enumerate(self.input_names):
            self.current_input[input_name] = mpc_optimal_action[input_i]

        self.history["inputs"].append(copy.deepcopy(self.current_input))
        self.history["references"].append(copy.deepcopy(self.current_reference))
        self.history["errors"].append(self._get_tracking_error(state))
        self.history["tvp"].append(self._tvp_data)

        return mpc_optimal_action

    # TODO: add support for changing Q and R in MPC and LQR

    def configure_viewer(self, viewer=None, plot_prediction=False):
        return super().configure_viewer(viewer=viewer, plot_prediction=False)

    def _get_p_values(self, t):
        for p_label in self._p_template.labels():
            p_name = p_label.split(",")[2]
            if p_name == "n_horizon":
                if len(self.history["mpc_horizon"]) == 0:
                    self._p_template["_p", 0, p_name] = self.mpc_config["params"]["n_horizon"]
                else:
                    self._p_template["_p", 0, p_name] = self.history["mpc_horizon"][-1]
            elif "_r" in p_name:
                self._p_template["_p", 0, p_name] = self._tvp_data[p_name][-1]

        return self._p_template


class TTAHMPC(AHMPC):
    def __init__(self, mpc_config, mpc_model=None, viewer=None):
        if "tvps" not in mpc_config["model"]:
            mpc_config["model"]["tvps"] = {}

        for comp in ["x", "y", "u_s"]:
            mpc_config["model"]["tvps"]["trajectory_{}".format(comp)] = {
                "true": [{
                    "redraw_probability": 0,
                    "type": "constant",
                    "kw": {
                        "value": 0
                    }
                }]
            }

        self.n_objects = mpc_config["params"].pop("n_objects", 0)
        if self.n_objects > 0:
            if "auxs" not in mpc_config["model"]:
                mpc_config["model"]["auxs"] = {}
            self.obj_kw = data = {"r": {"u_low": 0.5, "u_high": 1, "sigma": 0.05, "theta": 0.75, "low": 0.01, "high": 0.1}, "x": {"u_low": 0, "u_high": 0, "sigma": 0.5, "theta": 0.15, "low": 0.5, "high": 2}}
            data["y"] = data["x"]
            for obj_i in range(self.n_objects):
                for comp in ["x", "y", "r"]:
                    mpc_config["model"]["tvps"]["obj_{}_{}".format(obj_i, comp)] = {"true": [
                        {
                          "type": "uniform",
                          "forecast_aware": True,
                          "redraw_probability": 0,
                          "kw": {
                            "low": data[comp]["u_low"],
                            "high": data[comp]["u_high"]
                          }
                        }
                      ],
                      "forecast": [
                        {
                          "starts_at": 5,
                          "type": "OU",
                          "kw": {
                            "mean": 0,
                            "sigma": data[comp]["sigma"],
                            "theta": data[comp]["theta"],
                            "dt": mpc_config["params"]["t_step"],
                            "low": data[comp]["low"],
                            "high": data[comp]["high"]
                          }
                        }
                      ]
                    }
                mpc_config["model"]["auxs"]["obj_{}_distance".format(obj_i)] = {
                    "variables": [
                                     {"name": "obj_{}_{}".format(obj_i, comp),
                                      "type": "_tvp"} for comp in ["r", "x", "y"]
                                 ] + [
                                     {"name": state,
                                      "type": "_x"} for state in ["x", "y"]
                                 ],
                    "expression": "-(casadi.sqrt((obj_{}_x - x) ** 2 + (obj_{}_y - y) ** 2) - obj_{}_r)".format(obj_i, obj_i, obj_i),
                }
                for soft_constraint in [True, False]:
                    mpc_config["constraints"].append({
                        "variables": [{"name": "obj_{}_distance".format(obj_i), "type": "_aux"}] +
                                     ([{"name": "obj_{}_r".format(obj_i), "type": "_tvp"}] if soft_constraint else []),
                        "soft": soft_constraint,
                        "expression": "obj_{}_distance".format(obj_i) + ("+ 0.5 * obj_{}_r".format(obj_i) if soft_constraint else ""),
                        "cost": 1000,
                        "value": 0,
                        "aux": True,
                        "name": "obj_{}_distance-{}".format(obj_i, "s" if soft_constraint else "h")  # TODO: fix this shit better naming
                    })
        else:
            self.n_objects = 0
        super().__init__(mpc_config, mpc_model=mpc_model, viewer=viewer)
        self.trajectory = None
        self.u_s_ref = 3 * self.mpc_config["params"]["t_step"]
        self.extra_render_axes = 1
        self.goal_x = None
        self.goal_y = None
        self.obj_data = None
        self.object_noise_seed = None
        self._max_obj_dist = self.u_s_ref * 40
        self._max_horizon = 50
        self._n_noiseless = 0

    def reset(self, state=None, reference=None, constraint=None, tvp=None):
        self.obj_data = None
        if reference is not None and "ns" in reference:
            self.object_noise_seed = reference.pop("ns")
        else:
            self.object_noise_seed = []
            for obj_i in range(self.n_objects):
                self.object_noise_seed.append([])
                for comp in ["x", "y", "r"]:
                    self.object_noise_seed[obj_i].append(np.random.uniform(self.obj_kw[comp]["low"], self.obj_kw[comp]["high"]))
                    if np.random.uniform() < 0.5:
                        self.object_noise_seed[obj_i][-1] *= -1
        super().reset(state=state, reference=reference, constraint=constraint, tvp=tvp)

    def get_action(self, state, n_horizon, tvp_values=None):
        if self.obj_data is None:
            self.obj_data = []
            for obj_i in range(self.n_objects):
                self.obj_data.append({comp: np.full((self.mpc.n_horizon + 1,), tvp_values["obj_{}_{}".format(obj_i, comp)][0]) for comp in ["x", "y", "r"]})

        for obj_i in range(self.n_objects):
            rel_obj_dist = min(self.get_obj_distance(state, obj_i) / self._max_obj_dist, 1)
            for comp_i, comp in enumerate(["x", "y", "r"]):
                forecast_noise = np.zeros((self.mpc.n_horizon + 1,))
                if n_horizon > self._n_noiseless:
                    forecast_noise[self._n_noiseless:n_horizon] = np.linspace(0, self.object_noise_seed[obj_i][comp_i] * rel_obj_dist, self._max_horizon)[:n_horizon - self._n_noiseless]
                tvp_values["obj_{}_{}".format(obj_i, comp)] = self.obj_data[obj_i][comp] + forecast_noise
        return super().get_action(state, n_horizon, tvp_values)

    def get_obj_distance(self, state, obj_i):
        return np.linalg.norm(np.array([state["x"], state["y"]]) - np.array([self.obj_data[obj_i]["x"][0], self.obj_data[obj_i]["y"][0]]))

    def configure_viewer(self, viewer=None, plot_prediction=False):
        viewer = super().configure_viewer(viewer=viewer, plot_prediction=False)
        viewer["axes"]["trajectory"] = viewer["axis_objects"][len(viewer["axes"])]
        viewer["trajectory_lines"] = {}
        viewer["trajectory_lines"]["vehicle"] = viewer["axes"]["trajectory"].plot([], [], label="vehicle")[0]
        viewer["trajectory_lines"]["trajectory"] = viewer["axes"]["trajectory"].plot([], [], label="trajectory", linestyle="dashed")[0]
        viewer["obstacle_patches"] = []
        for obj_i in range(self.n_objects):
            viewer["obstacle_patches"].append(plt.Circle((0, 0), 1, color="g"))
            viewer["axes"]["trajectory"].add_patch(viewer["obstacle_patches"][obj_i])

        self.viewer = viewer
        return self.viewer

    def render(self, show=True):
        super().render(show=False)
        self.viewer["trajectory_lines"]["vehicle"].set_data(self.mpc.data["_x", "x"].ravel(), self.mpc.data["_x", "y"].ravel())
        self.viewer["trajectory_lines"]["trajectory"].set_data([step["trajectory_x"][0] for step in self.history["tvp"]],
                                                               [step["trajectory_y"][0] for step in self.history["tvp"]])
        for obj_i in range(self.n_objects):
            self.viewer["obstacle_patches"][obj_i].center = (self._tvp_data["obj_{}_x".format(obj_i)][0], self._tvp_data["obj_{}_y".format(obj_i)][0])
            self.viewer["obstacle_patches"][obj_i].radius = self._tvp_data["obj_{}_r".format(obj_i)][0]

        self.viewer["axes"]["trajectory"].relim()
        self.viewer["axes"]["trajectory"].autoscale_view()

        if show:
            self.viewer["fig"].show()

    def _get_p_values(self, t):
        self._p_template = super()._get_p_values(t)
        self._p_template["_p", 0, "goal_x"] = self.goal_x
        self._p_template["_p", 0, "goal_y"] = self.goal_y
        return self._p_template












