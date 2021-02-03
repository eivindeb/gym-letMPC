import numpy as np
import re
import tensorflow as tf
from casadi import *
import casadi.tools


def str_replace_whole_words(string, pattern, replace_with):
    return re.sub(r'\b{}\b'.format(pattern), str(replace_with), string)


class OrnsteinUhlenbeckProcess:
    def __init__(self, mean, sigma, theta=.15, dt=1e-2, initial_noise=None, low=None, high=None):
        def numpyify(arg):
            if arg is not None:
                if not isinstance(arg, np.ndarray):
                    return np.array(arg)
                else:
                    return arg
            return arg
        super().__init__()
        mean = numpyify(mean)
        sigma = numpyify(sigma)
        low = numpyify(low)
        high = numpyify(high)
        if low is not None:
            assert low.shape == mean.shape
        if high is not None:
            assert high.shape == mean.shape
        self._theta = theta
        self._mu = mean
        self._sigma = sigma
        self._dt = dt
        self._low = low
        self._high = high
        self.initial_noise = initial_noise
        self.noise_prev = None
        self.np_random = None
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed)

    def __call__(self) -> np.ndarray:
        noise = self.noise_prev + self._theta * (self._mu - self.noise_prev) * self._dt + \
                self._sigma * np.sqrt(self._dt) * self.np_random.normal(size=self._mu.shape)
        if self._low is not None or self._high is not None:
            noise = np.clip(noise, self._low, self._high)
        self.noise_prev = noise
        return noise

    def reset(self) -> None:
        """
        reset the Ornstein Uhlenbeck noise, to the initial position
        """
        self.noise_prev = self.initial_noise if self.initial_noise is not None else self._mu#np.zeros_like(self._mu)

    def __repr__(self) -> str:
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self._mu, self._sigma)



class TensorFlowEvaluator(casadi.Callback):
    def __init__(self, t_in, t_out, session, opts={}):
        """
          t_in: list of inputs (tensorflow placeholders)
          t_out: list of outputs (tensors dependeant on those placeholders)
          session: a tensorflow session
        """
        casadi.Callback.__init__(self)
        assert isinstance(t_in, list)
        self.t_in = t_in
        assert isinstance(t_out, list)
        self.t_out = t_out
        self.construct("TensorFlowEvaluator", opts)
        self.session = session
        self.refs = []
        self.enabled = False

    def get_n_in(self):
        return len(self.t_in)

    def get_n_out(self):
        return len(self.t_out)

    def get_sparsity_in(self, i):
        ret = []
        for dim in self.t_in[i].get_shape().as_list():
            ret.append(dim if dim is not None else 1)
        return casadi.Sparsity.dense(*ret)

    def get_sparsity_out(self, i):
        ret = []
        for dim in self.t_out[i].get_shape().as_list():
            ret.append(dim if dim is not None else 1)
        return casadi.Sparsity.dense(*ret)

    def eval(self, *args):
        # Associate each tensorflow input with the numerical argument passed by CasADi
        args = args[0]
        assert len(args) == len(self.t_in)
        feed_dict = {}
        for i, placeholder in enumerate(self.t_in):
            if not isinstance(args[i], np.ndarray):
                feed_dict[placeholder] = args[i].toarray()
            else:
                feed_dict[placeholder] = args[i]
        # Evaluate the tensorflow expressions
        ret = self.session.run(self.t_out, feed_dict=feed_dict)
        """
        if len(args) == 1:
            state = args[0]
            if not isinstance(state, np.ndarray):
                state = state.toarray()
            print("{} : {}".format(state, ret))
        """
        if not self.enabled:
            res = []
            for v in ret:
                res.append(np.zeros_like(v))
            ret = res
        return ret

    # Vanilla tensorflow offers just the reverse mode AD
    def has_reverse(self, nadj):
        return nadj == 1

    def get_reverse(self, nadj, name, inames, onames, opts):
        with self.session.graph.as_default():
            # Construct tensorflow placeholders for the reverse seeds
            adj_seed = [tf.placeholder(shape=self.sparsity_out(i).shape, dtype=tf.float32) for i in range(self.n_out())]
            # Construct the reverse tensorflow graph through 'gradients'
            grad = tf.gradients(self.t_out, self.t_in, grad_ys=adj_seed)
            # Create another TensorFlowEvaluator object
            callback = TensorFlowEvaluator(self.t_in+adj_seed, grad, self.session, opts)
            # Make sure you keep a reference to it
            self.refs.append(callback)

            # Package it in the nominal_in+nominal_out+adj_seed form that CasADi expects
            nominal_in = self.mx_in()
            nominal_out = self.mx_out()
            adj_seed = self.mx_out()
            return casadi.Function(name, nominal_in+nominal_out+adj_seed, callback.call(nominal_in+adj_seed), inames, onames)

    def set_enabled(self, status):
        self.enabled = status
        if len(self.refs) > 0:
            for ref in self.refs:
                ref.set_enabled(status)


class casadiNNVF:
    def __init__(self, layers=()):
        assert isinstance(layers, list) or isinstance(layers, tuple)
        self.layers = layers
        self.weights = None
        self.biases = None
        self.weights_num = None
        self.biases_num = None
        self.eval_VF = None

    def create_function(self, state, parameters):
        input_data_cat = vertcat(state, parameters).T
        blank = SX.sym('blank')

        # neuron_weights=SX.sym('neuron_weights')
        in_size = input_data_cat.shape[1]
        h_l_ws, h_l_bs = [], []

        relu_activation = Function("relu_f", [blank], [casadi.fmax(0, blank)])

        hidden_layer = input_data_cat
        for l_i, l_units in enumerate(self.layers):
            h_l_ws.append(casadi.tools.entry('hl_{}_weights'.format(l_i), sym=SX.sym('hl_{}_weights'.format(l_i), in_size, l_units)))
            h_l_bs.append(casadi.tools.entry('hl_{}_bias'.format(l_i), sym=SX.sym('hl_{}_bias'.format(l_i), l_units, 1)))

            hidden_layer = relu_activation(
                (hidden_layer @ h_l_ws[l_i].sym) + (DM.ones(hidden_layer.shape[0], l_units) @ diag(h_l_bs[l_i].sym)))
            in_size = hidden_layer.shape[1]

        output_layer_weights = casadi.tools.entry("ol_weights", sym=SX.sym('ol_weights', in_size, 1))
        output_layer_bias = casadi.tools.entry("ol_bias", sym=SX.sym('ol_bias', 1))
        output_layer = (hidden_layer @ output_layer_weights.sym) + (
                    DM.ones(hidden_layer.shape[0], 1) @ diag(output_layer_bias.sym))

        self.weights = casadi.tools.struct_symSX(h_l_ws + [output_layer_weights])
        self.biases = casadi.tools.struct_symSX(h_l_bs + [output_layer_bias])

        self.weights_num = np.zeros(self.weights.shape)
        self.biases_num = np.zeros(self.biases.shape)

        self.eval_VF = Function('neural_network_evaluation',
                        [state, parameters, self.weights, self.biases],
                        [output_layer])

    def set_weights_and_biases(self, weights, biases):
        self.weights_num = np.concatenate([w.flatten(order="F") for w in weights]).reshape(-1, 1)
        self.biases_num = np.concatenate([b.flatten(order="F") for b in biases]).reshape(-1, 1)

    def save_to_file(self, save_folder, name=""):
        np.save(os.path.join(save_folder, "{}_cnnvf_weights.npy".format(name)), self.weights_num)
        np.save(os.path.join(save_folder, "{}_cnnvf_biases.npy".format(name)), self.biases_num)

    def load_from_file(self, load_folder, name=""):
        self.weights_num = np.load(os.path.join(load_folder, "{}_cnnvf_weights.npy".format(name)))
        self.biases_num = np.load(os.path.join(load_folder, "{}_cnnvf_biases.npy".format(name)))

