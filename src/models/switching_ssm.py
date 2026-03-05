import numpy as np


class KalmanFilter:
    def __init__(self, A, C, Q, R, x0, P0):
        self.A = A
        self.C = C
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, y):
        S = self.C @ self.P @ self.C.T + self.R
        K = self.P @ self.C.T @ np.linalg.inv(S)
        y_pred = self.C @ self.x
        self.x = self.x + K @ (y - y_pred)
        self.P = (np.eye(len(self.x)) - K @ self.C) @ self.P

        likelihood = np.exp(
            -0.5 * (y - y_pred).T @ np.linalg.inv(S) @ (y - y_pred)
        ) / np.sqrt(np.linalg.det(2 * np.pi * S))

        return likelihood


class SwitchingStateSpaceModel:

    def __init__(self, filters, transition_matrix):
        self.filters = filters
        self.M = len(filters)
        self.T = transition_matrix
        self.regime_probs = np.ones(self.M) / self.M

    def step(self, y):

        likelihoods = np.zeros(self.M)

        for i, kf in enumerate(self.filters):
            kf.predict()
            likelihoods[i] = kf.update(y)

        prior = self.T.T @ self.regime_probs
        posterior = likelihoods * prior
        posterior = posterior / np.sum(posterior)

        self.regime_probs = posterior

        return posterior


def generate_synthetic_data(T=200):

    x = np.zeros(T)
    regimes = np.zeros(T)

    for t in range(1, T):

        if t < 70:
            regimes[t] = 0
            x[t] = 0.8 * x[t - 1] + np.random.normal(0, 0.5)

        elif t < 140:
            regimes[t] = 1
            x[t] = 1.2 * x[t - 1] + np.random.normal(0, 1)

        else:
            regimes[t] = 2
            x[t] = 1.5 * x[t - 1] + np.random.normal(0, 2)

    y = x + np.random.normal(0, 0.5, T)

    return y.reshape(-1, 1), regimes


def main():

    y, true_regimes = generate_synthetic_data()

    A1 = np.array([[0.8]])
    A2 = np.array([[1.2]])
    A3 = np.array([[1.5]])

    C = np.array([[1]])

    Q = np.array([[0.2]])
    R = np.array([[0.5]])

    x0 = np.array([0])
    P0 = np.eye(1)

    kf1 = KalmanFilter(A1, C, Q, R, x0.copy(), P0.copy())
    kf2 = KalmanFilter(A2, C, Q, R, x0.copy(), P0.copy())
    kf3 = KalmanFilter(A3, C, Q, R, x0.copy(), P0.copy())

    transition_matrix = np.array([
        [0.90, 0.05, 0.05],
        [0.05, 0.90, 0.05],
        [0.05, 0.05, 0.90]
    ])

    model = SwitchingStateSpaceModel(
        [kf1, kf2, kf3],
        transition_matrix
    )

    probs = []

    for obs in y:
        p = model.step(obs)
        probs.append(p)

    probs = np.array(probs)

    print("Final regime probabilities:")
    print(probs[-1])


if __name__ == "__main__":
    main()
