import numpy as np
import re


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