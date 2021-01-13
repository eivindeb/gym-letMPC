import do_mpc
import matplotlib.pyplot as plt
from gym_let_mpc.model import mpc_model_get_variable_names, initialize_mpc_model
from gym_let_mpc.utils import OrnsteinUhlenbeckProcess
import copy
import collections.abc
import matplotlib
import numpy as np


def initialize_simulator(model, config, tvp_fun=None):
    simulator = do_mpc.simulator.Simulator(model)
    simulator.set_param(**config["params"])

    if tvp_fun is not None:
        simulator.set_tvp_fun(lambda t: simulator.get_tvp_template())
        simulator.tvp_fun = tvp_fun

    if "uncertainty" in config:
        raise NotImplementedError  # TODO: if using robust MPC

    simulator.setup()

    return simulator


class TVP:
    def __init__(self, name, true_props, forecast_props=None):
        self.name = name
        self.values = []
        self.np_random = None
        self.generators = {"true": []}
        if forecast_props is not None:
            self.generators["forecast"] = []
        self.seed()
        for props, prop_name in zip([true_props, forecast_props], ["true", "forecast"]):
            if prop_name == "forecast" and forecast_props is None:
                continue
            for component in props:
                self.generators[prop_name].append({"redraw_probability": component.get("redraw_probability", 1),
                                                "forecast_aware": component.get("forecast_aware", True),
                                                "distribution": self.create_generator(component["type"], component["kw"])})

    def reset(self):
        for gen_type in ["true", "forecast"]:
            if gen_type in self.generators:
                for gen in self.generators[gen_type]:
                    if hasattr(gen["distribution"], "reset"):
                        gen["distribution"].reset()

        self.values = []

    def create_generator(self, type, gen_kw):
        if type == "OU":
            return OrnsteinUhlenbeckProcess(**gen_kw)
        if type == "constant":
            return lambda: gen_kw["value"]
        elif type in ["uniform", "normal"]:
            return lambda: getattr(self.np_random, type)(**gen_kw)
        else:
            raise NotImplementedError

    def generate_values(self, n_steps):
        for i in range(n_steps):
            step_val = {"true": [], "forecast": []}
            for gen_type in ["true", "forecast"]:
                if gen_type in self.generators:
                    for comp_i, component in enumerate(self.generators[gen_type]):
                        if len(self.values) == 0 or self.np_random.uniform() <= component["redraw_probability"]:
                            step_val[gen_type].append(component["distribution"]())
                        else:
                            step_val[gen_type].append(self.values[-1][gen_type][comp_i])

            self.values.append(step_val)

    def get_values(self, t_start, t_end=None, with_noise=False):
        if t_end is None:
            return sum(self.values[t_start]["true"])
        else:
            return [sum([self.values[t if self.generators["true"][gen_i]["forecast_aware"] else t_start]["true"][gen_i]
                         for gen_i in range(len(self.generators["true"]))])
                    + sum(self.values[t]["forecast"]) *
                    (min(t_end - t_start, (t - t_start) * 2)) / (t_end - t_start) * (1 if with_noise else 0)
                    for t in range(t_start, t_end)]

    def plot(self, axis=None):
        if axis is not None:
            plot_object = axis
        else:
            plot_object = plt
        x = np.arange(0, len(self.values))
        plot_object.plot(x, [v["true"] for v in self.values], label="true")
        if "forecast" in self.values[0]:
            plot_object.plot(x, [sum(v["true"]) + sum(v["forecast"]) for v in self.values], label="forecast", linestyle="dashed")
        if axis is None:
            plot_object.title(self.name)
            plot_object.legend()
            plt.show()

    def get_plot_lines(self, axis, forecast=True):
        lines = {}
        lines["true"] = axis.add_line(matplotlib.lines.Line2D([], [], label="true",
                            color=next(axis._get_lines.prop_cycler)["color"], linestyle="solid"))
        if "forecast" in self.generators and forecast:
            lines["forecast"] = axis.add_line(matplotlib.lines.Line2D([], [], label="forecast", color="orange",
                                                                      linestyle="dashed", marker="x", markersize=15))

        return lines

    def set_plot_data(self, lines, x=None, forecast=True):
        if x is None:
            x = np.arange(0, len(self.values))
        lines["true"].set_data(x, [sum(v["true"]) for v in self.values])
        if "forecast" in lines and forecast:
            lines["forecast"].set_data(x, [sum(v["forecast"]) for v in self.values])

    def seed(self, seed=None):
        if seed is None:
            seed = 0
        for type_i, gen_type in enumerate(["true", "forecast"]):
            if gen_type in self.generators:
                for gen_i, gen in enumerate(self.generators[gen_type]):
                    if hasattr(gen["distribution"], "seed"):
                        gen["distribution"].seed(seed + (type_i + 1) * gen_i)
        self.np_random = np.random.RandomState(seed)


class ControlSystem:
    def __init__(self, sim_config, controller):
        self.controller = controller
        self.tvps = {}
        if len(sim_config["model"].get("tvps", [])) > 0:
            for name, props in sim_config["model"]["tvps"].items():
                self.tvps[name] = TVP(name, props["true"], props.get("forecast", None))
            sim_tvp_fun = self._get_sim_model_tvp_values
        else:
            sim_tvp_fun = None

        if self.controller.tvp_props is not None:
            for name, props in self.controller.tvp_props.items():
                if props != "sim":
                    self.tvps[name] = TVP(name, props["true"], props.get("forecast", None))

        self.simulator = initialize_simulator(initialize_mpc_model(sim_config["model"]), sim_config, tvp_fun=sim_tvp_fun)
        if sim_tvp_fun is not None:
            self._tvp_template = self.simulator.get_tvp_template()
        else:
            self._tvp_template = None

        self.state_names = sorted(sim_config["model"]["states"].keys())
        self.current_state = {s: None for s in self.state_names}
        self.initial_state = {s: 0 for s in self.state_names} # TODO: initial state should default to provided values in config (or zero if not provided)

        self.raw_actions = {name: i for i, name in enumerate(sorted(sim_config["model"]["inputs"]))
                            if sim_config["model"]["inputs"][name].get("raw_action")}

        self.viewer = None
        self.render_n_axes = self.simulator.model.n_x + self.simulator.model.n_u
        if sim_config["render"].get("tvp", False):
            self.render_n_axes += len(self.tvps)
        self.np_random = None
        self.seed()
        self.history = None
        self.config = sim_config
        self._preset_process_noise = None
        self._process_noise_props = None
        self._step_count = None
        # TODO: estimator

    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed)
        for seed_i, tvp in enumerate(self.tvps.values()):
            tvp_seed = seed + seed_i if seed is not None else None
            tvp.seed(tvp_seed)

    def reset(self, state=None, reference=None, constraint=None, model=None, process_noise=None, tvp=None):
        self._step_count = 0
        if model is not None:
            if "plant" in model:
                self._update_model(model["plant"])
            if "controller" in model:
                self.controller.update_model(model["controller"])
        self.simulator.reset_history()
        if state is None:
            state = self.initial_state
        for state_name in self.current_state:
            if state_name in state:
                self.current_state[state_name] = state[state_name]
            else:
                self.current_state[state_name] = self.initial_state[state_name]
        self.simulator.x0 = self.get_state_vector(self.current_state)
        
        self._preset_process_noise = process_noise
        self._process_noise_props = {s_name: {} for s_name in self.state_names}

        if tvp is not None:
            for name, vals in tvp.items():
                self.tvps[name].values = vals
        elif self.tvps is not None:
            for tvp in self.tvps.values():
                tvp.reset()
                tvp.generate_values(1)

        self.controller.reset(self.current_state, reference, constraint)
        self.history = {"state": [copy.deepcopy(self.current_state)],
                        "process_noise": [np.zeros_like(self._get_process_noise())],
                        "tvp": []}

    def step(self, action):
        if len(self.tvps) > 0:
            for tvp in self.tvps.values():
                if len(tvp.values) < self._step_count + self.controller.mpc.n_horizon:
                    tvp.generate_values(self._step_count + self.controller.mpc.n_horizon - len(tvp.values))
            tvp_forecasts = {name: self.tvps[name].get_values(self._step_count,
                                                              self._step_count + self.controller.mpc.n_horizon,
                                                              with_noise=True)
                             for name in self.controller.tvp_props}
        else:
            tvp_forecasts = None
        control_input = self.controller.get_action(self.current_state, action, tvp_values=tvp_forecasts)
        for raw_input_i, input_vector_i in enumerate(self.raw_actions.values()):  # TODO: consider improving this
             control_input = np.insert(control_input, input_vector_i, action[raw_input_i], axis=0)
        step_process_noise = self._get_process_noise()
        step_state = self.simulator.make_step(control_input, w0=step_process_noise.reshape(-1, 1))
        self.current_state.update(self._get_state_dict(step_state))
        for state in self.current_state:
            clip = self.config["model"]["states"][state].get("clip", None)
            if clip is not None:
                self.current_state[state] = np.clip(self.current_state[state], clip[0], clip[1])
        self.simulator._x0.master = self.get_state_vector(self.current_state)  # TODO: remove this
        self.history["state"].append(copy.deepcopy(self.current_state))
        self.history["process_noise"].append(step_process_noise)
        self.history["tvp"].append({name: self.tvps[name].get_values(self._step_count + 1)
                                             for name in self.config["model"].get("tvps", {})})
        self._step_count += 1

    def _get_process_noise(self):
        process_noise = []
        for state_name in self.state_names:
            state = self.config["model"]["states"][state_name]
            if "W" in state:
                if self._preset_process_noise is not None and state_name in self._preset_process_noise:
                    ind = self._step_count % len(self._preset_process_noise[state_name])
                    noise = self._preset_process_noise[state_name][ind]
                else:
                    if state["W"]["type"] == "impulse":
                        if self._process_noise_props[state_name].get("cooldown", 0) <= 0 and \
                                self.np_random.uniform() < state["W"]["probability"]:
                            self._process_noise_props[state_name] = {"cooldown": state["W"].get("cooldown", 0)}
                            noise = state["W"]["value"]
                        else:
                            if "cooldown" in self._process_noise_props[state_name]:
                                self._process_noise_props[state_name]["cooldown"] -= 1
                            noise = 0
                    elif state["W"]["type"] == "step":
                        if self._process_noise_props[state_name].get("active", False):
                            noise = state["W"]["value"]
                        else:
                            self._process_noise_props[state_name]["active"] = state["W"]["activate_on"] >= self._step_count
                            noise = 0
                    else:
                        try:
                            noise = getattr(self.np_random, state["W"]["type"])(**state["W"]["kw"]) * state["W"]["scale"]
                        except AttributeError:
                            raise ValueError("Unexpected noise type {}".format(state["W"]["type"]))
                process_noise.append(noise)

        return np.array(process_noise)

    def _get_sim_model_tvp_values(self, t_now):
        if isinstance(t_now, np.ndarray):
            assert len(t_now.shape) == 1 and t_now.shape[0] == 1
            t_now = round(t_now[0])
        for i, name in enumerate(self.config["model"]["tvps"]):
            self._tvp_template[name, i] = self.tvps[name].get_values(t_now)

        return self._tvp_template

    def get_state_vector(self, state):
        """
        Get controller state vector (x) from state dictionary.
        :param state: (dict) dictionary of string state name keys with float state values.
        :returns (np.ndarray of (float)) state vector with order as defined in self.state_names
        """
        return np.array([state[s_name] for s_name in self.state_names])

    def _get_state_dict(self, state_vector):
        if len(state_vector.shape) == 2 and state_vector.shape[1] == 1:
            state_vector = state_vector.reshape((-1,))
        return {s_name: state_vector[s_i] for s_i, s_name in enumerate(self.state_names)}

    def get_constraint_distances(self, constraint_names="all"):
        constraint_distances = {}
        if hasattr(self.controller, "constraints"):
            for c_name, constraint in self.controller.constraints.items():
                if constraint_names == "all" or c_name in constraint_names:
                    if constraint["var_type"] == "_x":
                        constraint_distance = constraint["value"] - self.current_state[constraint["var_name"]]
                    elif constraint["var_type"] == "_u":
                        constraint_distance = constraint["value"] - self.controller.current_input[constraint["var_name"]]
                    else:
                        raise ValueError
                    if constraint["constraint_type"] == "upper":
                        constraint_distance *= -1
                    constraint_distances[c_name] = constraint_distance

        return constraint_distances

    def _update_model(self, model):
        def update_dict_recursively(d, u):
            for k, v in u.items():
                if isinstance(v, collections.abc.Mapping):
                    d[k] = update_dict_recursively(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        assert model["states"].keys() <= self.config["model"]["states"].keys()
        self.config["model"] = update_dict_recursively(self.config["model"], model)
        self.simulator = initialize_simulator(initialize_mpc_model(self.config["model"]), self.config)
        if self.viewer is not None and "sim_graphics" in self.viewer:
            self.viewer["sim_graphics"].data = self.simulator.data

    def configure_viewer(self, figure=None, axes=None):
        assert figure is None or axes is not None
        if figure is None:
            figure, axes = plt.subplots(self.render_n_axes, sharex=True, figsize=(9, 16))
        axes = {k: axes[i] for i, k in enumerate(mpc_model_get_variable_names(self.simulator.model, "_x") +
                                                 mpc_model_get_variable_names(self.simulator.model, "_u") +
                                                 (list(self.tvps.keys()) if self.config["render"].get("tvp", False)
                                                                        else []))}
        self.viewer = {
            "sim_graphics": do_mpc.graphics.Graphics(self.simulator.data),
            "figure": figure,
            "axes": axes
        }
        for state in mpc_model_get_variable_names(self.simulator.model, "_x"):
            self.viewer["sim_graphics"].add_line("_x", state, axis=axes[state], label=state)
            self.viewer["axes"][state].set_ylabel(state)

        for control_input in mpc_model_get_variable_names(self.simulator.model, "_u"):
            self.viewer["sim_graphics"].add_line("_u", control_input, axis=self.viewer["axes"][control_input],
                                                 label=control_input)
            self.viewer["axes"][control_input].set_ylabel(control_input)

        self.viewer["axes"][control_input].set_xlabel("time [s]")

        if hasattr(self.controller, "configure_viewer"):
            self.viewer = self.controller.configure_viewer(self.viewer)

        if self.config.get("render", {}).get("process_noise", False):
            self.viewer["process_noise_lines"] = {}
            for state_name, state_props in self.config["model"]["states"].items():
                if "W" in state_props:
                    self.viewer["process_noise_lines"][state_name] = self.viewer["axes"][state_name].add_line(
                        matplotlib.lines.Line2D([], [], label="W-{}".format(state_name),
                                color=next(self.viewer["axes"][state_name]._get_lines.prop_cycler)["color"],
                                linestyle="dotted")
                    )

        if self.config.get("render", {}).get("tvp", False):
            self.viewer["tvp_lines"] = {}
            for tvp_name, tvp_obj in self.tvps.items():
                self.viewer["tvp_lines"][tvp_name] = tvp_obj.get_plot_lines(self.viewer["axes"][tvp_name],
                                                                            forecast=tvp_name in self.controller.tvp_props)
                self.viewer["axes"][tvp_name].set_ylabel("TVP: {}".format(tvp_name))

    def render(self, figure=None, axes=None, return_viewer=False):
        if self.viewer is None:
            self.configure_viewer(figure, axes)
        self.controller.render(show=False)
        self.viewer["sim_graphics"].plot_results()
        if "process_noise_lines" in self.viewer:
            process_noise_data = np.array(self.history["process_noise"])
            x_data = np.array(range(process_noise_data.shape[0])) * self.config["params"]["t_step"]
            i = 0
            for s_name in self.state_names:
                if s_name in self.viewer["process_noise_lines"]:
                    self.viewer["process_noise_lines"][s_name].set_data(x_data, process_noise_data[:, i])
                    i += 1
        if "tvp_lines" in self.viewer:
            for tvp_name, tvp_obj in self.tvps.items():
                x_data = np.arange(len(tvp_obj.values)) * self.config["params"]["t_step"]
                tvp_obj.set_plot_data(self.viewer["tvp_lines"][tvp_name], x=x_data, forecast=False)
                self.viewer["axes"][tvp_name].relim()
                self.viewer["axes"][tvp_name].autoscale_view()
        self.viewer["sim_graphics"].reset_axes()
        if return_viewer:
            return self.viewer
        else:
            for axis in self.viewer["axes"].values():
                axis.legend()
            self.viewer["figure"].show()

