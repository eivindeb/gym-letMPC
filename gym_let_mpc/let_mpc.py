import gym
from gym.utils import seeding
import numpy as np
import json
from gym_let_mpc.simulator import ControlSystem
from gym_let_mpc.controllers import ETMPC, AHMPC, TTAHMPC, mpc_get_aux_value, mpc_get_algstate_value
import collections.abc
import matplotlib.pyplot as plt
from gym_let_mpc.utils import str_replace_whole_words, casadiNNVF
import copy
import gym_let_mpc


class LetMPCEnv(gym.Env):
    def __init__(self, config_path, config_kw=None):
        def set_config_attrs(parent, kws):
            for attr, val in kws.items():
                if isinstance(parent, dict) and attr not in parent:
                    parent[attr] = val
                elif isinstance(val, dict): #or isinstance(parent[attr], list):
                    set_config_attrs(parent[attr], val)
                else:
                    parent[attr] = val

        with open(config_path) as file_object:
            config = json.load(file_object)

        if config_kw is not None:
            set_config_attrs(config, config_kw)

        if config["mpc"]["model"] == "plant":
            config["mpc"]["model"] = copy.deepcopy(config["plant"]["model"])
        elif config["mpc"]["model"].get("parameters", None) == "plant":
            config["mpc"]["model"]["parameters"] = copy.deepcopy(config["plant"]["model"]["parameters"])

        if "lqr" in config:
            if config["lqr"]["model"] == "plant":
                config["lqr"]["model"] = copy.deepcopy(config["plant"]["model"])
            elif config["lqr"]["model"] == "mpc":
                config["lqr"]["model"] = copy.deepcopy(config["mpc"]["model"])
            elif config["lqr"]["model"].get("parameters", None) == "plant":
                config["lqr"]["model"]["parameters"] = copy.deepcopy(config["plant"]["model"]["parameters"])
            elif config["lqr"]["model"].get("parameters", None) == "mpc":
                config["lqr"]["model"]["parameters"] = copy.deepcopy(config["mpc"]["model"]["parameters"])

        self.config = config
        assert "max_steps" in self.config["environment"]
        self.max_steps = self.config["environment"]["max_steps"]

        assert "randomize" in self.config["environment"]
        assert "state" in self.config["environment"]["randomize"] and "reference" in self.config["environment"]["randomize"]
        assert "render" in self.config["environment"]
        if config["mpc"]["type"] in ["ETMPC", "ETMPCMIX", "AHETMPCMIX"]:
            #assert len(config["environment"]["action"]["variables"]) == 1 and \
            #       config["environment"]["action"]["variables"][0]["name"] == "mpc_compute"
            controller = getattr(gym_let_mpc.controllers, config["mpc"]["type"])(config["mpc"], config["lqr"], max_steps=self.max_steps)
            #self.action_space = gym.spaces.MultiBinary(1)
            a_low, a_high = [0], [1]
            if config["mpc"]["type"] == "AHETMPCMIX":
                a_low.append(1)
                a_high.append(config["mpc"]["params"]["n_horizon"])
            for a_i, a_name in enumerate(self.config["mpc"]["model"]["inputs"]):
                for c_name, c_props in controller.constraints.items():
                    if c_props["name"] == a_name:
                        if c_props["constraint_type"] == "lower":
                            a_low.append(c_props["value"] * 2)
                        else:
                            a_high.append(c_props["value"] * 2)
                if len(a_low) == a_i:
                    a_low.append(-np.finfo(np.float32).max)
                if len(a_high) == a_i:
                    a_high.append(np.finfo(np.float32).max)
            self.action_space = gym.spaces.Box(low=np.array(a_low, dtype=np.float32),
                                               high=np.array(a_high, dtype=np.float32),
                                               dtype=np.float32)
        elif config["mpc"]["type"] in ["AHMPC", "TTAHMPC", "TTAHMPCRANGE"]:
            assert len(config["environment"]["action"]["variables"]) == 1 and \
                   config["environment"]["action"]["variables"][0]["name"] == "mpc_horizon"
            if config["mpc"]["type"] == "AHMPC":
                controller = AHMPC(config["mpc"])
            elif config["mpc"]["type"] in ["TTAHMPC", "TTAHMPCRANGE"]:
                controller = getattr(gym_let_mpc.controllers, config["mpc"]["type"])(config["mpc"])
                if "end_on_constraint_violation" not in config["environment"]:
                    config["environment"]["end_on_constraint_violation"] = []
                for obj_i in range(controller.n_objects):
                    config["environment"]["end_on_constraint_violation"].append("obj_{}_distance".format(obj_i))
            else:
                raise ValueError
            self.action_space = gym.spaces.Box(low=np.array([1]), high=np.array([config["mpc"]["params"]["n_horizon"]]), dtype=np.float32)
        elif "LQR" in config["mpc"]["type"]:  # TODO: remove this test stuff
            controller = getattr(gym_let_mpc.controllers, config["mpc"]["type"])(config["mpc"], config["lqr"])
            a_low, a_high = [], []
            for a_i, a_name in enumerate(self.config["mpc"]["model"]["inputs"]):
                for c_name, c_props in controller.constraints.items():
                    if c_props["name"] == a_name:
                        if c_props["constraint_type"] == "lower":
                            a_low.append(c_props["value"])
                        else:
                            a_high.append(c_props["value"])
                if len(a_low) == a_i:
                    a_low.append(-np.finfo(np.float32).max)
                if len(a_high) == a_i:
                    a_high.append(np.finfo(np.float32).max)
            self.action_space = gym.spaces.Box(low=np.array(a_low, dtype=np.float32),
                                               high=np.array(a_high, dtype=np.float32),
                                               dtype=np.float32)
        else:
            raise ValueError
        self.control_system = ControlSystem(config["plant"], controller=controller)
        self.history = None
        self.steps_count = None
        self.np_random = None
        self.min_constraint_delta = 0.25  # TODO: how and where to set
        self.obs_module_idxs = []

        def get_observation_space(vars):
            obs_high = []
            obs_low = []
            for obs_var in vars:
                for var_transform in obs_var.get("transform", ["none"]):
                    self.obs_module_idxs.append(obs_var.get("module", "et"))
                    for lim_i, lim in enumerate(obs_var.get("limits", [None, None])):
                        if lim is None:
                            if lim_i == 0:
                                obs_low.append(-np.finfo(np.float32).max)
                            else:
                                obs_high.append(np.finfo(np.float32).max)
                        else:
                            if var_transform == "none":
                                if lim_i == 0:
                                    obs_low.append(lim)
                                else:
                                    obs_high.append(lim)
                            elif var_transform == "absolute":
                                if lim_i == 0:
                                    obs_low.append(0)
                                else:
                                    obs_high.append(lim)
                            elif var_transform == "square":
                                if lim_i == 0:
                                    obs_low.append(0)
                                else:
                                    obs_high.append(lim ** 2)
                            else:
                                raise NotImplementedError
            return obs_high, obs_low
        if isinstance(self.config["environment"]["observation"]["variables"], dict):
            obs_spaces = {}
            for obs_name, obs_vars in self.config["environment"]["observation"]["variables"].items():
                o_h, o_l = get_observation_space(obs_vars)
                obs_spaces[obs_name] = {"low": o_l, "high": o_l}
            self.observation_space = gym.spaces.Dict({k: gym.spaces.Box(low=np.array(v["low"], dtype=np.float32),
                                                                        high=np.array(v["high"], dtype=np.float32),
                                                                        dtype=np.float32) for k, v in obs_spaces.items()})
        else:
            obs_high, obs_low = get_observation_space(self.config["environment"]["observation"]["variables"])
            self.observation_space = gym.spaces.Box(low=np.array(obs_low, dtype=np.float32),
                                                    high=np.array(obs_high, dtype=np.float32),
                                                    dtype=np.float32)

        self.viewer = None

    def seed(self, seed=None):
        """
        Seed the random number generator of the control system.
        :param seed: (int) seed for random state
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.control_system.seed(seed)
        return [seed]

    def reset(self, state=None, reference=None, constraint=None, model=None, process_noise=None, tvp=None):
        """
        Reset state of environment. Note that the simulator is reset, the MPC solution is computed and the first
        MPC action is applied to the plant.

        :param state: (dict) initial conditions (value) for state name (key).
        :param reference: (dict) reference value (value) for reference name (key).
        :param constraint: (dict) constraint values (value) for constraint names (key).
        :param model: (dict) dictionary of dictionary where first key is model that it applies to ["plant", "mpc", "lqr"],
        first value is dictionary of model parameters where second value is the specified model parameter value.
        :param process_noise: (dict) process noise values (value) as ndarray for state name (key). The process noise at
        each time step loops through the provided array.
        :param tvp: (dict) values of time-varying parameters. New values are generated if values arent specified
        for all time steps elapsed.
        :return: ([float]) observation vector
        """
        def update_dict_recursively(d, u):
            for k, v in u.items():
                if isinstance(v, collections.abc.Mapping):
                    d[k] = update_dict_recursively(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        self.steps_count = 0

        sampled_state = self.sample_state()
        sampled_reference = self.sample_reference()
        sampled_constraint = self.sample_constraints()
        sampled_model = self.sample_model()

        if state is not None:
            sampled_state.update(state)
        elif len(sampled_state) == 0:
            sampled_state = None
        if reference is not None or (tvp is not None and any([k.endswith("_r") for k in tvp])):
            if reference is not None:
                reference.update({k: sum(v[0]["true"]) for k, v in tvp.items() if k.endswith("_r")})
            else:
                reference = {k: sum(v[0]["true"]) for k, v in tvp.items() if k.endswith("_r")}
            sampled_reference.update(reference)
        elif len(sampled_reference) == 0:
            sampled_reference = None
        if constraint is not None:
            sampled_constraint.update(constraint)
        elif len(sampled_constraint) == 0:
            sampled_constraint = None
        if model is not None:
            sampled_model = update_dict_recursively(sampled_model, model)
        elif len(sampled_model) == 0:
            sampled_model = None

        if self.config["mpc"]["type"] == "TTAHMPC":
            if tvp is None:
                tvp = {}
            for i, comp in enumerate(["x", "y", "theta"]):
                if comp not in sampled_state:
                    sampled_state[comp] = 0

            if sampled_reference is not None and "theta_r" in sampled_reference:
                self.theta_r = sampled_reference.pop("theta_r")
            else:
                self.theta_r = self.np_random.uniform(sampled_state["theta"] - np.radians(45), sampled_state["theta"] + np.radians(45))

            if sampled_reference is not None and "traj_steps" in sampled_reference:
                self.traj_steps = sampled_reference.pop("traj_steps")
            else:
                self.traj_steps = self.np_random.randint(60, 100)

            self.trajectory_start_x = 0
            self.trajectory_start_y = 0
            self.trajectory_goal_x = np.cos(self.theta_r) * self.traj_steps * self.control_system.controller.u_s_ref
            self.trajectory_goal_y = np.sin(self.theta_r) * self.traj_steps * self.control_system.controller.u_s_ref
            self.trajectory = {"x": np.linspace(sampled_state["x"], self.trajectory_goal_x, self.traj_steps),
                               "y": np.linspace(sampled_state["y"], self.trajectory_goal_y, self.traj_steps),
                               "u_s": np.full((self.traj_steps,), self.control_system.controller.u_s_ref)}
            for comp in self.trajectory:
                if "trajectory_{}".format(comp) not in tvp:
                    tvp["trajectory_{}".format(comp)] = [{"true": [v], "forecast": []} for v in self.trajectory[comp]]

            self.trajectory["n_steps"] = self.traj_steps
            self.control_system.controller.goal_x = self.trajectory_goal_x
            self.control_system.controller.goal_y = self.trajectory_goal_y

            self.traj_start = np.array([self.trajectory_start_x, self.trajectory_start_y])
            self.traj_goal = np.array([self.trajectory_goal_x, self.trajectory_goal_y])

            for obj_i in range(self.control_system.controller.n_objects):
                if "obj_{}_r".format(obj_i) not in tvp:
                    tvp.update({"obj_{}_{}".format(obj_i, comp): [{"forecast": [0]}] for comp in ["x", "y", "r"]})
                    obj_r = self.np_random.uniform(self.config["mpc"]["model"]["tvps"]["obj_0_r"]["true"][0]["kw"]["low"],
                                                                                  self.config["mpc"]["model"]["tvps"]["obj_0_r"]["true"][0]["kw"]["high"])
                    tvp["obj_{}_r".format(obj_i)][0]["true"] = [obj_r]

                    delta_x = self.np_random.uniform(-obj_r, obj_r)
                    delta_y = self.np_random.uniform(-np.sqrt(obj_r ** 2 - (delta_x ** 2)), np.sqrt(obj_r ** 2 - (delta_x ** 2)))
                    traj_xys = np.vstack((self.trajectory["x"], self.trajectory["y"])) + np.array([delta_x, delta_y]).reshape(-1, 1)
                    obj_feasible_points = [traj_i for traj_i in range(self.trajectory["n_steps"]) if
                                           np.linalg.norm(traj_xys[:, traj_i] - self.traj_start) > (obj_r * 1.75)
                                           and np.linalg.norm(traj_xys[:, traj_i] - self.traj_goal) > (obj_r * 1.75)]
                    obj_center_traj_i = self.np_random.choice(obj_feasible_points)
                    tvp["obj_{}_x".format(obj_i)][0]["true"] = [self.trajectory["x"][obj_center_traj_i] + delta_x]
                    tvp["obj_{}_y".format(obj_i)][0]["true"] = [self.trajectory["y"][obj_center_traj_i] + delta_y]

        self.control_system.reset(state=sampled_state, reference=sampled_reference, constraint=sampled_constraint,
                                  model=sampled_model, process_noise=process_noise, tvp=tvp)
        if self.config["mpc"]["type"] == "ETMPC":
            actions = {"mpc_compute": True}
            self.control_system.step(action=np.array([actions["mpc_compute"]]))
        elif self.config["mpc"]["type"] == "ETMPCMIX":
            actions = {"mpc_compute": True, "lqr": [0 for i in range(len(self.control_system.controller.input_names))]}
            self.control_system.step(action=np.array([actions["mpc_compute"] + actions["lqr"]]))
        elif self.config["mpc"]["type"] == "AHETMPCMIX":
            actions = {"mpc_compute": True, "mpc_horizon": self.config["mpc"]["params"]["n_horizon"], "lqr": [0 for i in range(len(self.control_system.controller.input_names))]}
            self.control_system.step(action=np.array([actions["mpc_compute"], actions["mpc_horizon"]] + actions["lqr"]))
        elif self.config["mpc"]["type"] in ["AHMPC", "TTAHMPC", "TTAHMPCRANGE"]:
            actions = {"mpc_horizon": self.config["mpc"]["params"]["n_horizon"]}
            self.control_system.step(action=np.array([actions["mpc_horizon"]]))
        elif self.config["mpc"]["type"] in ["LQRETMPC", "LQRFHFNMPC"]:  # TODO remove test
            actions = {"lqr": [0 for i in range(len(self.control_system.controller.input_names))]}
            self.control_system.step(action=np.array(actions["lqr"]))
        elif self.config["mpc"]["type"] == "LQRMPC":
            actions = {"lqr": [0 for i in range(len(self.control_system.controller.input_names))]}
            self.control_system.step(action=np.array(actions["lqr"]))
        #self.history["actions"].append(actions)
        obs = self.get_observation()
        self.history = {"obs": [obs], "actions": [], "rewards": []}

        return obs

    def step(self, action):
        assert not np.any(np.isnan(action))
        if isinstance(action, np.ndarray) and len(action.shape) > 1:
            action = action[0, :]
        if self.config["mpc"]["type"] == "ETMPC":
            a_dict = {"mpc_compute": action[0]}#round(action)}
            action = a_dict["mpc_compute"]
        elif self.config["mpc"]["type"] == "ETMPCMIX":
            a_dict = {"mpc_compute": action.flat[0], "lqr_noise": action.flat[1:]}
        elif self.config["mpc"]["type"] == "AHETMPCMIX":
            a_dict = {"mpc_compute": action.flat[0], "mpc_horizon": action.flat[1], "lqr_noise": action.flat[2:]}
        elif self.config["mpc"]["type"] in ["AHMPC", "TTAHMPC", "TTAHMPCRANGE"]:
            #action = np.clip(action, -1, 1)
            #action = np.array(50 - 1) * (action - (-1)) / (1 - (-1)) + 1
            a_dict = {a_props["name"]: np.round(action).astype(np.int32)
                      for a_i, a_props in enumerate(self.config["environment"]["action"]["variables"])}
            action = a_dict["mpc_horizon"]
        elif self.config["mpc"]["type"] in ["LQRMPC", "LQRFHFNMPC"]:  # TODO: remove this test stuff
            a_dict = {"lqr": action[0]}
        elif self.config["mpc"]["type"] == "LQRETMPC":
            a_dict = {"mpc_compute": False, "lqr": action}#{"mpc_compute": action[0], "lqr": action[1:]}
            #action = np.atleast_1d(action)
        else:
            raise ValueError
        self.control_system.step(action)#np.atleast_1d(int(a_dict["mpc_compute"])))
        info  = {}
        #if self.config["mpc"]["type"] in ["LQRETMPC", "ETMPC"] and self.control_system.controller.steps_since_mpc_computation >= 10:
        #    self.control_system.controller.get_action(self.control_system.current_state, np.array([True]))
        self.history["actions"].append(a_dict)
        self.steps_count += 1

       # info = {}
        obs = self.get_observation()
        done = False
        additional_rew = 0
        if self.steps_count >= self.max_steps:
            done = True
            info["termination"] = "steps"
        elif len(self.config["environment"].get("end_on_constraint_violation", [])) > 0:
            if isinstance(self.control_system.controller, TTAHMPC):
                for obj_i in range(self.control_system.controller.n_objects):
                    if self.control_system.controller.get_obj_distance(self.control_system.current_state, obj_i) <= 0:
                        done = True
                        info["termination"] = "constraint"
                        additional_rew = self.config["environment"]["reward"].get("termination_weight", -1) * (
                                    self.max_steps - self.steps_count)
                        info["reward/constraint"] = -additional_rew
                        break
            else:
                for c_name, c_d in self.control_system.get_constraint_distances().items():
                    if (c_name.startswith("clin-") or (c_name.startswith("cnlin-") and c_name.endswith("h"))) and\
                            c_name.split("-")[1] in self.config["environment"]["end_on_constraint_violation"] and c_d > 0:
                        done = True
                        info["termination"] = "constraint"
                        additional_rew = self.config["environment"]["reward"].get("termination_weight", -1) * (self.max_steps - self.steps_count)
                        info["reward/constraint"] = -additional_rew
                        break
        if info.get("termination", None) != "constraint":
            info["reward/constraint"] = 0
        if self.config["mpc"]["type"] == "TTAHMPC" and \
                np.linalg.norm(np.array([self.trajectory_goal_x, self.trajectory_goal_y]) - \
                               np.array([self.control_system.current_state["x"], self.control_system.current_state["y"]])) <= 0.5:
            done = True
            info["termination"] = "goal"

        rew = self.get_reward(done=done, info=info)
        rew += additional_rew
        for category, v in self.config["environment"].get("info", {}).items():
            if category == "reward":
                for rew_name, rew_expr in v.items():
                    info["reward/{}".format(rew_name)] = self.get_reward(rew_expr, done=done, info=info)
            else:
                raise NotImplementedError

        #info.update(a_dict)

        if self.control_system.controller.mpc_config["objective"].get("vf", None) is not None:
            mpc = self.control_system.controller.get_mpc()
            use_p_idxs = []
            for p_i, p_l in enumerate(mpc.model._p.labels()):
                if "n_horizon" not in p_l:
                    use_p_idxs.append(p_i)
            parameters = mpc.opt_p_num["_p", 0][use_p_idxs].toarray().ravel() if len(use_p_idxs) > 0 else []
            step_vf_data = {"mpc_state": np.concatenate([mpc.opt_p_num["_x0"].toarray().ravel(), parameters], axis=0),
                            "mpc_next_state": np.concatenate([self.control_system.get_state_vector(self.control_system.current_state), parameters], axis=0)}
            step_vf_data["mpc_n_horizon"] = self.control_system.controller.history["mpc_horizon"][-1]
            if self.config["plant"]["model"].get("auxs", {}).get("energy_kinetic", None) is not None and False:
                step_vf_data["mpc_rewards"] = mpc_get_aux_value(mpc, "energy_kinetic", 1) - mpc_get_aux_value(mpc, "energy_potential", 1)
            else:
                #step_vf_data["mpc_parameter"] -=
                step_vf_data["mpc_rewards"] = mpc.lterm_fun(self.control_system.get_state_vector(self.control_system.current_state),
                                                            mpc.opt_x_num_unscaled["_u", 0, 0],
                                                            mpc.opt_x_num_unscaled["_z", 1, 0, -1],
                                                            mpc.opt_p_num["_tvp", 0],
                                                            mpc.opt_p_num["_p", 0]).__float__()
            if mpc.use_nn_vf:
                info["mpc_value_fn"] = mpc.vf_fun(mpc.opt_x_num_unscaled["_x", -1, 0, -1],
                                                     parameters,
                                                     mpc.opt_p_num["_vf_weights"],
                                                     mpc.opt_p_num["_vf_biases"]).__float__()
            else:
                info["mpc_value_fn"] = mpc.mterm_fun(mpc.opt_x_num_unscaled["_x", -1, 0, -1], mpc.opt_p_num["_p", 0]).__float__()

            info["data"] = step_vf_data
            info["mpc_avg_stage_cost"] = step_vf_data["mpc_rewards"]

        #info["mpc_computation_time"] = sum([v for k, v in self.control_system.controller.get_mpc().solver_stats.items() if k.startswith("t_proc")])
        try:
            info["execution_time"] = self.control_system.controller.history["execution_time"][-1]
        except (KeyError, IndexError):
            info["execution_time"] = np.nan

        info.update({k: v.astype(np.float64) if hasattr(v, "dtype") else v for k, v in a_dict.items()})
        #if self.config["mpc"]["type"] == "LQRETMPC":
        #    info["action"] = self.control_system.controller.history["u_cfs"][-1]

        if False and self.config["mpc"]["type"] in ["ETMPC", "LQRETMPC"]:
            info["mpc_action"] = np.squeeze(self.control_system.controller.history["u_mpc"][-1], axis=0)
            if self.control_system.controller.steps_since_mpc_computation > 0:  # get unconstrained action  # TODO: how to handle end of horizon recompute?
                info["action"] = info["mpc_action"] + np.squeeze(self.control_system.controller.history["u_lqr"][-1], axis=0)
            else:
                if not a_dict["mpc_compute"] and not any(self.control_system.controller.history["mpc_compute"][-10:-1]):  # TODO: change to horizon
                    info["eoh_flag"] = True
                #elif not a_dict["mpc_compute"]:
                #    print("what")
                info["action"] = info["mpc_action"]
        if self.config["mpc"]["type"] in ["ETMPCMIX", "AHETMPCMIX", "LQRETMPC", "LQRFHFNMPC"] and self.control_system.controller.lqr_config["type"] == "time-varying":
            if self.control_system.controller.steps_since_mpc_computation == 0:
                As, Bs = self.get_linearized_mpc_model_over_prediction()
                info["As"] = As
                info["Bs"] = Bs
        if self.config["mpc"]["type"] == "LQRMPC" and self.config["lqr"].get("type", "time-invariant") == "time-varying":
            info["As"] = self.control_system.controller.lqr.A
            info["Bs"] = self.control_system.controller.lqr.B

        self.history["obs"].append(obs)
        self.history["rewards"].append(rew)

        return obs, rew, done, info

    def render(self, mode='human', save_path=None):  # TODO: add env renders
        figure, axes = None, None
        if self.viewer is None:
            env_plots = [plot_name for plot_name, make_plot in self.config["environment"]["render"].items() if make_plot]
            if len(env_plots) > 0:
                figure, axes = plt.subplots(self.control_system.render_n_axes + len(env_plots), sharex=False,
                                            figsize=(9, 16))
            self.viewer = self.control_system.render(figure=figure, axes=axes, return_viewer=True)
            for i, plot in enumerate(env_plots):
                self.viewer["axes"][plot] = axes[-(i + 1)]
        else:
            self.viewer = self.control_system.render(figure=figure, axes=axes, return_viewer=True)
        for plot_name, make_plot in self.config["environment"]["render"].items():
            if make_plot:
                self.viewer["axes"][plot_name].set_ylabel("-".join(plot_name.split("_")[1:]))
                x_data = np.array(range(self.steps_count)) * self.control_system.config["params"]["t_step"]
                self.viewer["axes"][plot_name].clear()
                if plot_name == "plot_action":
                    for a_var in self.config["environment"]["action"]["variables"]:
                        y_data = [step_a[a_var["name"]] for step_a in self.history["actions"]]
                        self.viewer["axes"][plot_name].plot(x_data, y_data, label=a_var["name"], drawstyle="steps")
                elif plot_name == "plot_reward":
                    self.viewer["axes"][plot_name].plot(x_data, self.history["rewards"], label="reward")
                    self.viewer["axes"][plot_name].text(max(x_data) + self.control_system.config["params"]["t_step"],
                                                        self.history["rewards"][-1],
                                                        "{:.3f}".format(np.sum(self.history["rewards"])))
                else:
                    raise ValueError
        for axis in self.viewer["axes"].values():
            axis.legend()
        if save_path is not None:
            self.viewer["figure"].savefig(save_path, bbox_inches="tight", format="png")
            plt.close(self.viewer["figure"])
        else:
            self.viewer["figure"].show()

    def get_observation(self):
        def _transform_val(_val, _transform):
            if _transform == "none":
                return _val
            elif _transform == "absolute":
                return abs(_val)
            elif _transform == "square":
                return _val ** 2
            elif _transform == "sqrt":
                return np.sqrt(_val)
            else:
                raise ValueError
            
        if isinstance(self.config["environment"]["observation"]["variables"], dict):
            obs = {}
            for obs_name, vars in self.config["environment"]["observation"]["variables"].items():
                obs[obs_name] = []
                for var in vars:
                    var_val = self._get_variable_value(var)
                    for transform in var.get("transform", ["none"]):
                        obs[obs_name].append(_transform_val(var_val, transform))
                obs[obs_name] = np.array(obs[obs_name])
        else:
            obs = []
            for var in self.config["environment"]["observation"]["variables"]:
                var_val = self._get_variable_value(var)
                for transform in var.get("transform", ["none"]):
                    obs.append(_transform_val(var_val, transform))
            obs = np.array(obs)

        return obs

    def get_reward(self, rew_expr=None, done=False, info=None):
        if rew_expr is None:
            rew_expr = self.config["environment"]["reward"]["expression"]

        rew_expr = str_replace_whole_words(rew_expr, "done", int(done))

        for var in sorted(self.config["environment"]["reward"]["variables"], key=lambda x: len(x), reverse=True):
            var_val = self._get_variable_value(var)
            if isinstance(var_val, list) or isinstance(var_val, np.ndarray):  # TODO: needs to be better way to do this
                var_val = var_val[0]
            rew_expr = str_replace_whole_words(rew_expr, var["name"], var_val)

        return eval(rew_expr)

    def _get_variable_value(self, var):
        if var["type"] == "state":
            val = self.control_system.current_state[var["name"]]
        elif var["type"] == "mpc_state":
            if var.get("value_type", "mpc_computation"):
                val = self.control_system.history["state"][-(self.control_system.controller.steps_since_mpc_computation + 2)][var["name"]]
            else:
                raise ValueError
        elif var["type"] == "input":
            if var.get("value_type", "absolute") == "absolute":
                val = self.control_system.controller.current_input[var["name"]]
            elif var.get("value_type") == "delta":
                val = self.control_system.controller.history["inputs"][-2][var["name"]] - \
                      self.control_system.controller.current_input[var["name"]]
            else:
                raise ValueError
        elif var["type"] == "algstate":
            val = mpc_get_algstate_value(self.control_system.controller.get_mpc(), var["name"])
        elif var["type"] == "reference":
            val = self.control_system.controller.current_reference[var["name"]]
        elif var["type"] == "tvp":
            val = self.control_system.tvps[var["name"]].get_values(self.steps_count + var.get("index", 0))
        elif var["type"] == "error":
            val = self.control_system.controller.history["errors"][-1][var["name"]]
            if np.isnan(val):
                val = 0
        elif var["type"] == "eps": # TODO: remove (test stuff)
            val = self.control_system.current_state[var["name"]] - self.control_system.controller._tvp_data[var["name"] + "_r"][0]
        elif var["type"] == "epsilon":  #TODO: ensure newest epsilon (i.e. from applying recompute/not recompute)
            if var.get("t", None) is not None:
                val = self.control_system.controller.history["epsilons"][-var.get("t")][var["name"]]
            else:
                if var["name"] in self.control_system.state_names:
                    if self.control_system.controller.steps_since_mpc_computation + 1 < self.control_system.controller.mpc_state_preds.shape[1]:
                        pred = self.control_system.controller.mpc_state_preds[:, self.control_system.controller.steps_since_mpc_computation + 1, :]
                    else:
                        pred = self.control_system.controller.get_lqr_steady_state().reshape(-1, 1)
                    eps_vec = self.control_system.get_state_vector(self.control_system.current_state).reshape(-1, 1) - pred
                    val = self.control_system.get_state_dict(eps_vec)[var["name"]]
                else:
                    val = self.control_system.controller.history["epsilons"][-1][var["name"]]
            if np.isnan(val):
                val = 0
        elif var["type"] == "constraint":
            if var.get("value_type") == "distance":
                val = self.control_system.get_constraint_distances((var["name"],))[var["name"]]
            else:
                raise ValueError
        elif var["type"] == "action":
            if var.get("value_type", "agent") == "agent":
                val = self.history["actions"][-1][var["name"]]
            elif var.get("value_type", "agent") == "controller":
                val = self.control_system.controller.history[var["name"]][-1]
            elif var.get("value_type", "agent") == "mpc_computation":
                val = self.control_system.controller.history[var["name"]][-(self.control_system.controller.steps_since_mpc_computation + 1)]
            else:
                raise ValueError
        elif var["type"] == "time":
            if var.get("value_type") == "fraction":
                val = self.control_system.controller.steps_since_mpc_computation / self.control_system.controller.get_mpc().n_horizon
            elif var.get("value_type") == "absolute":  # TODO: Is this correct? it seems like it caps perhaps?
                val = self.control_system.controller.steps_since_mpc_computation
            else:
                raise ValueError
        elif var["type"] == "parameter":
            if var["value_type"] in ["plant", "mpc", "lqr"]:
                val = self.config[var["value_type"]]["model"]["parameters"][var["name"]]
            else:
                raise ValueError
        elif var["type"] == "aux":
            val = mpc_get_aux_value(self.control_system.controller.get_mpc(), var["name"])
        elif var["type"] == "obj_dist":
            val = self.control_system.controller.get_obj_distance(self.control_system.current_state, int(var["name"].split("_")[1]))
        else:
            raise ValueError

        if isinstance(val, np.ndarray):
            val = val[0]
        if "limits" in var:
            val = np.clip(val, var["limits"][0], var["limits"][1])

        return val

    def sample_constraints(self):
        constraints = {}
        for c_name, c_props in self.config["environment"].get("randomize", {}).get("constraints", {}).items():
            constraint_val = getattr(self.np_random, c_props["type"])(**c_props["kw"])
            if c_name.split("-")[1] in [k.split("-")[1] for k in constraints.keys()]:
                other_bound_type = "u" if c_name.split("-")[2] == "l" else "l"
                other_bound_val = constraints[c_name[:-1] + other_bound_type]
                if other_bound_type == "u":
                    constraint_val = min(other_bound_val - self.min_constraint_delta, constraint_val)
                else:
                    constraint_val = max(other_bound_val + self.min_constraint_delta, constraint_val)
            constraints[c_name] = constraint_val
        return constraints

    def sample_state(self):
        state = {}
        for s_name, s_props in self.config["environment"].get("randomize", {}).get("state", {}).items():
            if s_props["type"] == "constant":
                state[s_name] = s_props["value"]
            else:
                state[s_name] = getattr(self.np_random, s_props["type"])(**s_props["kw"])

        return state

    def sample_reference(self):
        reference = {}
        for r_name, r_props in self.config["environment"].get("randomize", {}).get("reference", {}).items():
            reference[r_name] = getattr(self.np_random, r_props["type"])(**r_props["kw"])

        return reference

    def sample_model(self):
        model = {}
        for s_name, s_props in self.config["environment"].get("randomize", {}).get("model", {}).get("states", {}).items():
            model["states"] = {s_name: {}}
            for component_name, component_props in s_props.items():
                model["states"][s_name][component_name] = \
                    {comp_v_name: getattr(self.np_random, v_prop["type"])(**v_prop["kw"])
                                    for comp_v_name, v_prop in component_props.items()}

        model = {dest: model for dest in self.config["environment"].get("randomize", {}).get("model", {}).get("apply", [])}
        return model

    def stop(self):
        pass

    def create_dataset(self, n_scenarios):
        dataset = []
        self.reset()
        for i in range(n_scenarios):
            process_noise = np.array([self.control_system._get_process_noise() for i in range(self.max_steps)])
            ep_dict = {"state": self.sample_state(), "reference": self.sample_reference(),
                            "constraint": self.sample_constraints(), "model": self.sample_model(),
                       "process_noise": {}, "tvp": {}}
            if self.config["mpc"]["type"] == "TTAHMPC":
                ep_dict["reference"].update({"theta_r": self.theta_r, "traj_steps": self.traj_steps, "ns": self.control_system.controller.object_noise_seed})
            s_i = 0
            for s_name, s_props in self.config["plant"]["model"]["states"].items():
                if "W" in s_props:
                    ep_dict["process_noise"][s_name] = process_noise[:, s_i]
                    s_i += 1
            for tvp_name, tvp_obj in self.control_system.tvps.items():
                tvp_obj.generate_values(self.max_steps)
                ep_dict["tvp"][tvp_name] = tvp_obj.values
            dataset.append(ep_dict)
            self.reset()

        return dataset

    def set_value_function_weights_and_biases(self, weights, biases):
        self.control_system.controller.set_value_function_weights_and_biases(weights, biases)

    def save_value_function(self, save_folder, name=""):
        self.control_system.controller.save_value_function(save_folder, name=name)

    def load_value_function(self, load_folder, name=""):
        self.control_system.controller.load_value_function(load_folder, name=name)

    def set_value_function_enabled_status(self, status):
        """
        Enable/disable value function (i.e. if enabled it replaces mterm in the cost function).
        :param status: enbabled/disabled
        :return:
        """
        assert self.control_system.controller.mpc_config["objective"].get("vf", None) is not None
        self.control_system.controller.update_mpc_params({"use_nn_vf": status})

    def get_lqr(self):
        return self.control_system.controller.lqr

    def update_lqr(self, **components):
        assert hasattr(self.control_system.controller, "lqr")
        self.control_system.controller.lqr.update_component(**components)

    def set_action_noise_properties(self, properties):
        if isinstance(properties, dict):
            for u_name, u_kw in properties.items():
                self.control_system.controller.mpc_config["model"]["inputs"][u_name]["noise"]["kw"].update(u_kw)
        elif isinstance(properties, list):
            for u_i, u_name in enumerate(self.control_system.controller.input_names):
                self.control_system.controller.mpc_config["model"]["inputs"][u_name]["noise"]["kw"].update(properties[u_i])
        else:
            raise ValueError("Properties must either be a dictionary with key input name and value input kws, or a list of input kws where order is taken from controller.input_names")

    def get_linearized_mpc_model_over_prediction(self):  # TODO: should depend on n_horizon (when using AHMPC with weights)
        if self.config["mpc"]["type"] == "LQRMPC" and self.config["lqr"].get("type", "time-invariant") == "time-varying":  # TODO: remove (test stuff)
            As, Bs = [], []
            x1_r = self.control_system.controller._tvp_data["x1_r"][1:]
            for i in range(10):
                system = self.control_system.controller.mpc.get_linearized_model_at(np.array([x1_r[i], 0]), np.array([0]))
                As.append(system[0])
                Bs.append(system[1])
            return As, Bs
        else:
            return self.control_system.controller.mpc.get_linearized_model_over_prediction()


if __name__ == "__main__":  # TODO: constraints on pendulum and end episode if constraints violated
    env = LetMPCEnv("configs/cart_pendulum_horizon.json")#"../../lmpc-horizon/configs/unicycle_ca_horizon.json")
    env.seed(5)

    """
    from tensorflow_casadi import TensorFlowEvaluator, MLP
    import tensorflow as tf
    a = tf.placeholder(shape=(None, 4), dtype=tf.float32)
    mlp = MLP(a)
    sess = tf.Session()
    val_fun = TensorFlowEvaluator([mlp.input_ph], [mlp.output], sess)
    env.set_value_function(mlp.input_ph, mlp.output, sess)
    """
    test_set_path = "../../lmpc-horizon/datasets/cart_pendulum_10.pkl"#"../../lmpc-horizon/datasets/unicycle_ca_5.pkl"

    import pickle
    with open(test_set_path, "rb") as f:
        test_set = pickle.load(f)

    rews = {}

    for i in range(1):
        import time
        obs = env.reset(**test_set[5])
        #obs = env.reset(state={"pos": 0, "omega": -1, "theta": -0.53, "v": 0})
        #obs = env.reset()

        done = False
        t_before = time.process_time()
        horizon = 15
        while not done:
            t_step = time.process_time()
            if env.steps_count % 1 == 0 and False:
                horizon = 25 if horizon == 50 else 50
            obs, rew, done, info = env.step([horizon])#[np.random.randint(1, 10)])
            for rew_comp, v in info.items():
                if rew_comp.startswith("reward/"):
                    if rew_comp not in rews:
                        rews[rew_comp] = []
                    rews[rew_comp].append(v)
            if time.process_time() - t_step > 1:
                print(env.control_system.controller.mpc.solver_stats)
            print(env.steps_count)

        for k, v in rews.items():
            print("{}: {}".format(k, sum(v)))
        print("Elapsed time {}".format(time.process_time() - t_before))
        print("Termination: {}".format(info["termination"]))
        env.render()


        



