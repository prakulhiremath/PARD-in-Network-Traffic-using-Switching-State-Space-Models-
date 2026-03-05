"""
run_switching.py
----------------
Main experiment script: run the Switching SSM on preprocessed data.

Usage:
    python experiments/run_switching.py --data data/processed/cicids2017_features.npy \
                                        --labels data/processed/cicids2017_labels.npy \
                                        --regimes 4 \
                                        --state-dim 8 \
                                        --em-iters 15
"""

import argparse
import sys
import os
import numpy as np
import pickle
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing.feature_engineering import build_observation_sequence, temporal_train_test_split
from src.inference.variational_switching import SwitchingSSM
from src.utils.metrics import full_evaluation_report
from src.utils.visualization import (
    plot_regime_probabilities,
    plot_hidden_state,
    plot_confusion_matrix,
    plot_log_likelihood_curve,
    plot_detection_timeline,
)


def run_experiment(args):
    print("\n" + "=" * 60)
    print("PARD-SSM: Probabilistic Attack Regime Detection")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"\n[1/5] Loading data from {args.data}")
    X = np.load(args.data)
    y = np.load(args.labels)
    print(f"      Features: {X.shape}, Labels: {y.shape}")
    print(f"      Regimes found: {sorted(np.unique(y).tolist())}")

    # ------------------------------------------------------------------
    # 2. Preprocess
    # ------------------------------------------------------------------
    print(f"\n[2/5] Preprocessing features...")
    X_train, X_test, y_train, y_test = temporal_train_test_split(X, y, test_ratio=0.2)

    result = build_observation_sequence(
        X_train, y_train,
        normalize=True,
        n_pca_components=args.obs_dim,
        window_size=args.window_size,
    )

    result_test = build_observation_sequence(
        X_test, y_test,
        normalize=False,  # use training scaler in production
        n_pca_components=args.obs_dim,
        window_size=args.window_size,
    )

    obs_train = result["observations"]    # (W_train, T, obs_dim)
    obs_test = result_test["observations"]

    print(f"      Train windows: {obs_train.shape}")
    print(f"      Test windows : {obs_test.shape}")

    # Flatten windows for sequence model (use first window for demo)
    # In full training: iterate over all windows
    train_seq = obs_train[0]   # (T, obs_dim)
    test_seq = obs_test[0]
    y_test_seq = result_test["labels"][0]

    # ------------------------------------------------------------------
    # 3. Build and train Switching SSM
    # ------------------------------------------------------------------
    print(f"\n[3/5] Training Switching SSM...")
    print(f"      n_regimes={args.regimes}, state_dim={args.state_dim}, obs_dim={args.obs_dim}")

    model = SwitchingSSM(
        n_regimes=args.regimes,
        state_dim=args.state_dim,
        obs_dim=args.obs_dim,
    )

    log_likelihoods = model.fit(train_seq, n_iter=args.em_iters, verbose=True)

    # ------------------------------------------------------------------
    # 4. Inference on test set
    # ------------------------------------------------------------------
    print(f"\n[4/5] Running inference on test sequence...")
    inference_result = model.filter(test_seq)

    print(f"      Test log-likelihood: {inference_result.log_likelihood:.2f}")

    # ------------------------------------------------------------------
    # 5. Evaluate
    # ------------------------------------------------------------------
    print(f"\n[5/5] Evaluation...")

    predicted_labels = inference_result.viterbi_path
    true_labels = y_test_seq[:len(predicted_labels)]

    # Truncate to min length
    min_len = min(len(predicted_labels), len(true_labels))
    predicted_labels = predicted_labels[:min_len]
    true_labels = true_labels[:min_len]
    regime_probs = inference_result.regime_probs[:min_len]
    predicted_obs = inference_result.predicted_observations[:min_len]

    metrics = full_evaluation_report(
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        regime_probs=regime_probs,
        observations=test_seq[:min_len],
        predicted_observations=predicted_obs,
        log_likelihood=inference_result.log_likelihood,
        attack_regimes=list(range(1, args.regimes)),
    )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    os.makedirs("experiments/results", exist_ok=True)
    output_path = "experiments/results/switching_output.pkl"
    with open(output_path, "wb") as f:
        pickle.dump({
            "metrics": metrics,
            "regime_probs": regime_probs,
            "viterbi_path": predicted_labels,
            "true_labels": true_labels,
            "log_likelihoods": log_likelihoods,
            "filtered_means": inference_result.filtered_means,
        }, f)
    print(f"\nResults saved to {output_path}")

    # ------------------------------------------------------------------
    # Visualize
    # ------------------------------------------------------------------
    if not args.no_plot:
        print("\nGenerating plots...")
        plot_log_likelihood_curve(log_likelihoods)
        plot_regime_probabilities(
            regime_probs,
            true_labels=true_labels,
            viterbi_path=predicted_labels,
        )
        plot_hidden_state(inference_result.filtered_means)
        plot_confusion_matrix(true_labels, predicted_labels)
        plot_detection_timeline(
            regime_probs,
            true_labels,
            attack_regime_ids=list(range(1, args.regimes)),
        )

    print("\nDone!")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Switching SSM experiment")
    parser.add_argument("--data", required=True, help="Path to features .npy file")
    parser.add_argument("--labels", required=True, help="Path to labels .npy file")
    parser.add_argument("--regimes", type=int, default=4, help="Number of attack regimes")
    parser.add_argument("--state-dim", type=int, default=8, help="Hidden state dimension")
    parser.add_argument("--obs-dim", type=int, default=10, help="Observation dim (PCA)")
    parser.add_argument("--window-size", type=int, default=100, help="Time window length")
    parser.add_argument("--em-iters", type=int, default=15, help="EM iterations")
    parser.add_argument("--no-plot", action="store_true", help="Skip plots")
    args = parser.parse_args()

    run_experiment(args)
