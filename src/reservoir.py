import numpy as np


class EchoStateNetwork:
    def __init__(
        self,
        input_dim=None,
        reservoir_size=300,
        spectral_radius=0.9,
        input_scale=0.5,
        leak_rate=1.0,
        density=0.1,
        ridge_alpha=1e-6,
        seed=1,
    ):
        self.input_dim = input_dim
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.input_scale = input_scale
        self.leak_rate = leak_rate
        self.density = density
        self.ridge_alpha = ridge_alpha
        self.seed = seed

        self.w_in = None
        self.w = None
        self.w_out = None

        if input_dim is not None:
            self._initialize(input_dim)

    def _initialize(self, input_dim):
        self.input_dim = input_dim
        rng = np.random.default_rng(self.seed)

        self.w_in = rng.uniform(
            low=-self.input_scale,
            high=self.input_scale,
            size=(self.reservoir_size, self.input_dim + 1),
        )

        w = rng.uniform(
            low=-1.0,
            high=1.0,
            size=(self.reservoir_size, self.reservoir_size),
        )

        mask = rng.random((self.reservoir_size, self.reservoir_size)) < self.density
        w = w * mask

        radius = np.max(np.abs(np.linalg.eigvals(w)))

        if radius > 0:
            w = w * (self.spectral_radius / radius)

        self.w = w.real

    def _state_from_sequence(self, sequence):
        state = np.zeros(self.reservoir_size, dtype=float)

        for row in sequence:
            input_vector = np.concatenate(([1.0], row))
            pre_activation = self.w @ state + self.w_in @ input_vector
            updated_state = np.tanh(pre_activation)
            state = (1.0 - self.leak_rate) * state + self.leak_rate * updated_state

        return state

    def transform(self, x):
        x = np.asarray(x, dtype=float)

        if x.ndim != 3:
            raise ValueError("Input must have shape: samples x time x features")

        if self.input_dim is None:
            self._initialize(x.shape[2])

        if x.shape[2] != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x.shape[2]}")

        states = np.zeros(
            (x.shape[0], 1 + self.reservoir_size + self.input_dim),
            dtype=float,
        )

        for i in range(x.shape[0]):
            state = self._state_from_sequence(x[i])
            last_input = x[i, -1]
            states[i] = np.concatenate(([1.0], state, last_input))

        return states

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        states = self.transform(x)

        identity = np.eye(states.shape[1])
        identity[0, 0] = 0.0

        left = states.T @ states + self.ridge_alpha * identity
        right = states.T @ y

        try:
            self.w_out = np.linalg.solve(left, right)
        except np.linalg.LinAlgError:
            self.w_out = np.linalg.pinv(left) @ right

        return self

    def decision_function(self, x):
        if self.w_out is None:
            raise ValueError("Model has not been fitted.")

        states = self.transform(x)
        return states @ self.w_out

    def predict_proba(self, x):
        scores = self.decision_function(x)
        return np.clip(scores, 0.0, 1.0)

    def predict(self, x, threshold=0.5):
        risks = self.predict_proba(x)
        return (risks >= threshold).astype(int)