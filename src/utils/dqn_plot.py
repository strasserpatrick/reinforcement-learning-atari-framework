import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_metrics(data_path=None):
    data = pd.read_csv(data_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Total Reward per epoch
    sns.lineplot(x="epoch", y="validation_reward_epoch", data=data, ax=ax1)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation Reward Epoch")
    ax1.set_title("Validation Reward Epoch")

    # Plot 2: Loss per epoch
    sns.lineplot(x="epoch", y="loss", data=data, ax=ax2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Loss per epoch")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_metrics()
