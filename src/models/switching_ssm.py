import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from src.utils.visualization import plot_regime_probabilities


# -------------------- Kalman Filter --------------------
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
        # FIXED BUG
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, y):

        S = self.C @ self.P @ self.C.T + self.R
        S += np.eye(S.shape[0]) * 1e-6

        K = self.P @ self.C.T @ np.linalg.inv(S)
        y_pred = self.C @ self.x

        self.x = self.x + K @ (y - y_pred)
        self.P = (np.eye(self.P.shape[0]) - K @ self.C) @ self.P

        diff = y - y_pred
        exponent = -0.5 * diff.T @ np.linalg.inv(S) @ diff

        det_S = max(np.linalg.det(S), 1e-6)
        dim = y.shape[0]
        denom = np.sqrt(((2 * np.pi) ** dim) * det_S)

        likelihood = np.exp(exponent.item()) / denom
        return likelihood


# -------------------- Switching Model --------------------
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

        posterior = likelihoods * prior + 1e-8
        posterior = posterior / np.sum(posterior)

        self.regime_probs = posterior
        return posterior


# -------------------- MAIN --------------------
def main():

    # Load dataset
    X = np.load("data/processed/cicids2017_features.npy")
    y = np.load("data/processed/cicids2017_labels.npy")

    # -------------------- DATASET INFO --------------------
    print("\nDataset Info:")
    print("Total samples loaded:", X.shape[0])

    # Optional: enforce using only 50k samples
    MAX_SAMPLES = 50000
    if X.shape[0] > MAX_SAMPLES:
        X = X[:MAX_SAMPLES]
        y = y[:MAX_SAMPLES]
        print(f"Using subset of {MAX_SAMPLES} samples for this experiment")

    print("Final dataset size used:", X.shape[0])

    # -------------------- FEATURE SELECTION --------------------
    # RF features (rich)
    selector = SelectKBest(score_func=f_classif, k='all')
    rf_features = selector.fit_transform(X, y)

    # SSSM features (low-dimensional)
    sssm_input = X[:, :5]

    # Normalize
    rf_features = (rf_features - np.mean(rf_features, axis=0)) / (np.std(rf_features, axis=0) + 1e-6)
    sssm_input = (sssm_input - np.mean(sssm_input, axis=0)) / (np.std(sssm_input, axis=0) + 1e-6)

    # Labels
    true_regimes = (y != 0).astype(int)

    # -------------------- TEMPORAL SPLIT --------------------
    n = len(rf_features)
    split = int(0.8 * n)
    print("\nData Split:")
    print("Training samples:", split)
    print("Testing samples:", n - split)

    rf_train = rf_features[:split]
    rf_test = rf_features[split:]

    true_train = true_regimes[:split]
    true_test = true_regimes[split:]

    # -------------------- MODEL INIT --------------------
    dim = sssm_input.shape[1]

    # Stronger regime separation
    A1 = np.eye(dim) * 0.3
    A2 = np.eye(dim) * 2.0

    C = np.eye(dim)

    Q1 = np.eye(dim) * 0.001
    Q2 = np.eye(dim) * 0.1

    R = np.eye(dim) * 0.2

    x0 = np.zeros(dim)
    P0 = np.eye(dim)

    kf1 = KalmanFilter(A1, C, Q1, R, x0.copy(), P0.copy())
    kf2 = KalmanFilter(A2, C, Q2, R, x0.copy(), P0.copy())

    transition_matrix = np.array([
        [0.97, 0.03],
        [0.03, 0.97]
    ])

    model = SwitchingStateSpaceModel([kf1, kf2], transition_matrix)

    # -------------------- RUN SSSM --------------------
    regime_probs_history = []

    for obs in sssm_input:
        p = model.step(obs)
        regime_probs_history.append(p)

    regime_probs_history = np.array(regime_probs_history)
    # -------- SMOOTHING (VERY IMPORTANT) --------
    alpha = 0.8
    for i in range(1, len(regime_probs_history)):
        regime_probs_history[i] = (
            alpha * regime_probs_history[i]
            + (1 - alpha) * regime_probs_history[i - 1]
        )

    # -------------------- TEMPORAL FEATURES --------------------
    diff_features = np.diff(rf_features, axis=0, prepend=rf_features[0:1])

    # Combine all features
    X_combined = np.hstack([rf_features, diff_features, regime_probs_history])

    # -------------------- TRAIN-TEST SPLIT --------------------
    X_train = X_combined[:split]
    X_test = X_combined[split:]

    y_train = true_regimes[:split]
    y_test = true_regimes[split:]

    # -------------------- OVERSAMPLING --------------------
    X_minority = X_train[y_train == 1]
    y_minority = y_train[y_train == 1]

    X_majority = X_train[y_train == 0]
    y_majority = y_train[y_train == 0]

    X_minority_upsampled, y_minority_upsampled = resample(
        X_minority, y_minority,
        replace=True,
        n_samples=len(X_majority),
        random_state=42
    )

    X_train_balanced = np.vstack([X_majority, X_minority_upsampled])
    y_train_balanced = np.hstack([y_majority, y_minority_upsampled])

    # -------------------- RANDOM FOREST --------------------
    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        min_samples_split=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )

    clf.fit(X_train_balanced, y_train_balanced)
    from src.explainability.shap_explainer import run_shap_analysis
    run_shap_analysis(clf, X_train_balanced, X_test, rf_features.shape[1])

    # -------------------- TEST EVALUATION --------------------
    y_pred = clf.predict(X_test)
    y_probs = clf.predict_proba(X_test)[:, 1]   # probability of class "Attack"

    print("\n HYBRID MODEL RESULTS (TEST SET):")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # -------------------- ROC CURVE --------------------
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    print(f"ROC-AUC Score: {roc_auc:.4f}")

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle='--')  # random line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    plt.show()

    # -------------------- PRECISION-RECALL CURVE --------------------
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    pr_auc = average_precision_score(y_test, y_probs)

    print(f"PR-AUC Score: {pr_auc:.4f}")

    plt.figure()
    plt.plot(recall, precision, label=f"PR Curve (AP = {pr_auc:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid()
    plt.show()

    # -------------------- CONFUSION MATRIX (TEST) --------------------
    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix (Test Set):")
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Attack"])
    disp.plot(cmap="Blues", values_format='d')
    plt.title("Confusion Matrix - Test Set")
    plt.show()

    # -------------------- FULL DATASET (OPTIONAL) --------------------
    y_pred_full = clf.predict(X_combined)

    print("\n HYBRID MODEL RESULTS (FULL DATASET):")
    print("Accuracy:", accuracy_score(true_regimes, y_pred_full))
    print(classification_report(true_regimes, y_pred_full))

    # -------------------- CONFUSION MATRIX (FULL DATASET) --------------------
    cm_full = confusion_matrix(true_regimes, y_pred_full)

    print("\nConfusion Matrix (Full Dataset):")
    print(cm_full)

    disp_full = ConfusionMatrixDisplay(confusion_matrix=cm_full, display_labels=["Normal", "Attack"])
    disp_full.plot(cmap="Blues", values_format='d')
    plt.title("Confusion Matrix - Full Dataset")
    plt.show()

    # -------------------- SSSM EVALUATION --------------------
    predicted_regimes = np.argmax(regime_probs_history, axis=1)
    predicted_test = predicted_regimes[split:]

    print("\nSSSM Accuracy:", accuracy_score(true_test, predicted_test))

    # -------------------- SHORT WINDOW VISUALIZATION --------------------
    window_size = 300
    start = split   # start from test region
    end = start + window_size

    attack_probs = regime_probs_history[start:end, 1]

    plt.figure()
    plt.plot(attack_probs)
    plt.title("Attack Regime Probability (Short Window)")
    plt.xlabel("Time Step")
    plt.ylabel("Probability")
    plt.grid()
    plt.show()

    # -------------------- VISUALIZATION --------------------
    plot_regime_probabilities(regime_probs_history)


if __name__ == "__main__":
    main()
