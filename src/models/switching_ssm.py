import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from src.utils.visualization import plot_regime_probabilities


# ==================== Kalman Filter ====================
class KalmanFilter:

    def __init__(self, A, C, Q, R, x0, P0):
        self.A  = A
        self.C  = C
        self.Q  = Q
        self.R  = R
        self.x  = x0.copy()   # always copy — never mutate shared initial state
        self.P  = P0.copy()

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, y):
        S  = self.C @ self.P @ self.C.T + self.R
        S += np.eye(S.shape[0]) * 1e-6            # regularise to avoid singular S

        K      = self.P @ self.C.T @ np.linalg.inv(S)
        y_pred = self.C @ self.x
        diff   = y - y_pred

        self.x = self.x + K @ diff

        # Joseph form — numerically stable covariance update
        I_KC   = np.eye(self.P.shape[0]) - K @ self.C
        self.P = I_KC @ self.P @ I_KC.T + K @ self.R @ K.T

        # Log-likelihood — avoids underflow to 0.0 in high-dim / large-det cases
        sign, log_det_S = np.linalg.slogdet(S)
        log_det_S       = max(log_det_S, -1e6)
        dim             = y.shape[0]
        log_like        = (
            -0.5 * float(diff.T @ np.linalg.inv(S) @ diff)
            - 0.5 * log_det_S
            - 0.5 * dim * np.log(2 * np.pi)
        )
        log_like = np.clip(log_like, -500, 0)
        return np.exp(log_like)

    def reset(self, x0, P0):
        self.x = x0.copy()
        self.P = P0.copy()


# ==================== Switching State-Space Model ====================
class SwitchingStateSpaceModel:

    def __init__(self, filters, transition_matrix, prior=None):
        self.filters = filters
        self.M       = len(filters)
        self.T       = transition_matrix
        # Allow a custom prior (e.g. matching class imbalance in the dataset)
        self.regime_probs = prior if prior is not None else np.ones(self.M) / self.M

    def step(self, y):
        likelihoods = np.zeros(self.M)
        for i, kf in enumerate(self.filters):
            kf.predict()
            likelihoods[i] = kf.update(y)

        prior     = self.T.T @ self.regime_probs
        posterior = likelihoods * prior + 1e-8
        posterior = posterior / posterior.sum()

        self.regime_probs = posterior
        return posterior.copy()


# ==================== Helpers ====================
def causal_smooth(probs, alpha, seed=None):
    """
    Forward-only (causal) exponential smoothing.
    seed: last row from a preceding window, to avoid a cold-start discontinuity.
    """
    smoothed = probs.copy().astype(float)
    start_row = seed if seed is not None else smoothed[0]
    smoothed[0] = alpha * smoothed[0] + (1 - alpha) * start_row
    for i in range(1, len(smoothed)):
        smoothed[i] = alpha * smoothed[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


# ==================== MAIN ====================
def main():

    # ---------- Load ----------
    X = np.load("data/processed/cicids2017_features.npy")
    y = np.load("data/processed/cicids2017_labels.npy")

    print("\nDataset Info:")
    print("Total samples loaded:", X.shape[0])

    MAX_SAMPLES = 50_000
    if X.shape[0] > MAX_SAMPLES:
        X = X[:MAX_SAMPLES]
        y = y[:MAX_SAMPLES]
        print(f"Using subset of {MAX_SAMPLES} samples")

    print("Final dataset size:", X.shape[0])

    true_regimes = (y != 0).astype(int)

    # ---------- Temporal split FIRST (no leakage) ----------
    n     = len(X)
    split = int(0.8 * n)
    print(f"\nData split — train: {split}, test: {n - split}")

    X_tr, X_te       = X[:split],            X[split:]
    y_tr, y_te       = true_regimes[:split],  true_regimes[split:]
    attack_ratio      = y_tr.mean()
    print(f"Attack ratio in training set: {attack_ratio:.3f}")

    # ---------- Feature selection (fit on train only) ----------
    n_features = X_tr.shape[1]
    # FIX: k must not exceed n_features — choose the lesser
    k_select   = min(20, n_features)
    print(f"SelectKBest k={k_select} (dataset has {n_features} features)")

    selector = SelectKBest(score_func=f_classif, k=k_select)
    selector.fit(X_tr, y_tr)

    rf_tr_raw = selector.transform(X_tr)
    rf_te_raw = selector.transform(X_te)

    # Normalise with train statistics only
    rf_mean = rf_tr_raw.mean(axis=0)
    rf_std  = rf_tr_raw.std(axis=0) + 1e-6
    rf_tr   = (rf_tr_raw - rf_mean) / rf_std
    rf_te   = (rf_te_raw - rf_mean) / rf_std

    rf_dim = rf_tr.shape[1]   # actual number of selected features

    # ---------- SSSM input: top-min(5,n_features) by ANOVA score ----------
    k_sssm     = min(5, n_features)
    top_idx    = np.argsort(selector.scores_)[::-1][:k_sssm]
    sssm_tr_raw = X_tr[:, top_idx]
    sssm_te_raw = X_te[:, top_idx]

    sssm_mean = sssm_tr_raw.mean(axis=0)
    sssm_std  = sssm_tr_raw.std(axis=0) + 1e-6
    sssm_tr   = (sssm_tr_raw - sssm_mean) / sssm_std
    sssm_te   = (sssm_te_raw - sssm_mean) / sssm_std

    dim = sssm_tr.shape[1]
    print(f"SSSM input dimension: {dim}")

    # ---------- Kalman filter setup ----------
    # Normal regime: slow, low-noise dynamics
    A1 = np.eye(dim) * 0.3
    Q1 = np.eye(dim) * 0.001

    # Attack regime: faster, higher-noise dynamics
    A2 = np.eye(dim) * 1.5
    Q2 = np.eye(dim) * 0.5    # higher process noise → more responsive to attacks

    C  = np.eye(dim)
    R  = np.eye(dim) * 0.2
    x0 = np.zeros(dim)
    P0 = np.eye(dim)

    kf1 = KalmanFilter(A1, C, Q1, R, x0, P0)
    kf2 = KalmanFilter(A2, C, Q2, R, x0, P0)

    # Transition matrix: tuned to match dataset attack ratio
    # Lower self-transition than 0.97 so the model switches more readily
    p_stay_normal = 0.90
    p_stay_attack = 0.85   # attack bouts are often shorter — exit faster
    transition_matrix = np.array([
        [p_stay_normal,      1 - p_stay_normal],
        [1 - p_stay_attack,  p_stay_attack    ],
    ])

    # Informative prior: weight initial regime probs by class frequency
    prior = np.array([1 - attack_ratio, attack_ratio])
    model = SwitchingStateSpaceModel([kf1, kf2], transition_matrix, prior=prior)

    # ---------- SSSM — train window ----------
    sssm_probs_tr = []
    for obs in sssm_tr:
        sssm_probs_tr.append(model.step(obs))
    sssm_probs_tr = np.array(sssm_probs_tr)

    # ---------- SSSM — test window (filter state continues causally) ----------
    sssm_probs_te = []
    for obs in sssm_te:
        sssm_probs_te.append(model.step(obs))
    sssm_probs_te = np.array(sssm_probs_te)

    # ---------- Causal smoothing (no future leakage into train features) ----------
    alpha          = 0.7                              # slightly lower → sharper peaks
    smooth_tr      = causal_smooth(sssm_probs_tr, alpha)
    smooth_te      = causal_smooth(sssm_probs_te, alpha, seed=smooth_tr[-1])
    smooth_history = np.vstack([smooth_tr, smooth_te])

    # ---------- SSSM standalone accuracy ----------
    sssm_pred_tr = np.argmax(smooth_tr, axis=1)
    sssm_pred_te = np.argmax(smooth_te, axis=1)
    print(f"\nSSSM accuracy (train): {accuracy_score(y_tr, sssm_pred_tr):.4f}")
    print(f"SSSM accuracy (test):  {accuracy_score(y_te, sssm_pred_te):.4f}")

    # ---------- Temporal diff features ----------
    diff_tr = np.diff(rf_tr, axis=0, prepend=rf_tr[0:1])
    diff_te = np.diff(rf_te, axis=0, prepend=rf_te[0:1])

    # ---------- Assemble combined feature matrices ----------
    X_train_comb = np.hstack([rf_tr, diff_tr, smooth_tr])
    X_test_comb  = np.hstack([rf_te, diff_te, smooth_te])

    # ---------- Oversampling (train only, new-style RNG) ----------
    rng        = np.random.default_rng(42)
    mask_min   = y_tr == 1
    mask_maj   = y_tr == 0
    X_min, y_min = X_train_comb[mask_min], y_tr[mask_min]
    X_maj, y_maj = X_train_comb[mask_maj], y_tr[mask_maj]

    idx_up           = rng.choice(len(X_min), size=len(X_maj), replace=True)
    X_min_up         = X_min[idx_up]
    y_min_up         = y_min[idx_up]
    X_train_balanced = np.vstack([X_maj, X_min_up])
    y_train_balanced = np.hstack([y_maj, y_min_up])

    # ---------- Random Forest ----------
    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        min_samples_split=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train_balanced, y_train_balanced)

    # SHAP analysis
    from src.explainability.shap_explainer import run_shap_analysis
    run_shap_analysis(clf, X_train_balanced, X_test_comb, rf_dim)

    # ---------- Test evaluation (default threshold 0.5) ----------
    y_pred  = clf.predict(X_test_comb)
    y_probs = clf.predict_proba(X_test_comb)[:, 1]

    print("\nHYBRID MODEL RESULTS (TEST SET — threshold 0.5):")
    print("Accuracy:", accuracy_score(y_te, y_pred))
    print(classification_report(y_te, y_pred, target_names=["Normal", "Attack"]))

    # ---------- Lower threshold for higher attack recall ----------
    threshold    = 0.40
    y_pred_40    = (y_probs >= threshold).astype(int)
    print(f"HYBRID MODEL RESULTS (TEST SET — threshold {threshold}):")
    print("Accuracy:", accuracy_score(y_te, y_pred_40))
    print(classification_report(y_te, y_pred_40, target_names=["Normal", "Attack"]))

    # ---------- ROC ----------
    fpr, tpr, _ = roc_curve(y_te, y_probs)
    roc_auc     = auc(fpr, tpr)
    print(f"ROC-AUC: {roc_auc:.4f}")

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(); plt.grid(); plt.tight_layout(); plt.show()

    # ---------- PR ----------
    precision, recall, _ = precision_recall_curve(y_te, y_probs)
    pr_auc               = average_precision_score(y_te, y_probs)
    print(f"PR-AUC: {pr_auc:.4f}")

    plt.figure()
    plt.plot(recall, precision, label=f"PR Curve (AP = {pr_auc:.4f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(); plt.grid(); plt.tight_layout(); plt.show()

    # ---------- Confusion matrix — test ----------
    cm   = confusion_matrix(y_te, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Attack"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix — Test Set"); plt.tight_layout(); plt.show()

    # ---------- Full dataset ----------
    X_full       = np.vstack([X_train_comb, X_test_comb])
    y_pred_full  = clf.predict(X_full)

    print("\nHYBRID MODEL RESULTS (FULL DATASET):")
    print("Accuracy:", accuracy_score(true_regimes, y_pred_full))
    print(classification_report(true_regimes, y_pred_full, target_names=["Normal", "Attack"]))

    cm_full   = confusion_matrix(true_regimes, y_pred_full)
    disp_full = ConfusionMatrixDisplay(cm_full, display_labels=["Normal", "Attack"])
    disp_full.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix — Full Dataset"); plt.tight_layout(); plt.show()

    # ---------- Short-window attack probability plot ----------
    window = 300
    attack_probs_window = smooth_history[split: split + window, 1]
    plt.figure()
    plt.plot(attack_probs_window)
    plt.axhline(0.5, color="r", linestyle="--", alpha=0.5, label="Decision boundary")
    plt.title("Attack Regime Probability (Short Window)")
    plt.xlabel("Time Step"); plt.ylabel("Probability")
    plt.legend(); plt.grid(); plt.tight_layout(); plt.show()

    # ---------- Full regime probability visualisation ----------
    plot_regime_probabilities(smooth_history)


if __name__ == "__main__":
    main()
