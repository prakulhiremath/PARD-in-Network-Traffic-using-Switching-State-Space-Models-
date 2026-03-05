import numpy as np
import matplotlib.pyplot as plt
import imageio
import os


def plot_regime_probabilities(probs, save_path="regime_probabilities.png"):

    plt.figure(figsize=(10,5))

    plt.plot(probs[:,0], label="Normal")
    plt.plot(probs[:,1], label="Scanning")
    plt.plot(probs[:,2], label="Intrusion")

    plt.xlabel("Time")
    plt.ylabel("Probability")
    plt.title("Attack Regime Probabilities Over Time")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def generate_regime_gif(probs, output_gif="docs/regime_detection.gif"):

    os.makedirs("frames", exist_ok=True)

    frames = []

    for t in range(len(probs)):

        plt.figure(figsize=(8,4))

        plt.plot(probs[:t+1,0], label="Normal")
        plt.plot(probs[:t+1,1], label="Scanning")
        plt.plot(probs[:t+1,2], label="Intrusion")

        plt.xlabel("Time")
        plt.ylabel("Probability")
        plt.title("Attack Regime Detection")

        plt.ylim(0,1)
        plt.legend()
        plt.grid(True)

        frame_path = f"frames/frame_{t}.png"
        plt.savefig(frame_path)
        plt.close()

        frames.append(imageio.imread(frame_path))

    os.makedirs("docs", exist_ok=True)

    imageio.mimsave(output_gif, frames, duration=0.1)

    print(f"GIF saved to {output_gif}")


def demo():

    T = 200

    probs = np.zeros((T,3))

    for t in range(T):

        if t < 70:
            probs[t] = [0.9,0.1,0.0]

        elif t < 140:
            probs[t] = [0.1,0.8,0.1]

        else:
            probs[t] = [0.05,0.15,0.8]

    plot_regime_probabilities(probs)

    generate_regime_gif(probs)


if __name__ == "__main__":
    demo()
