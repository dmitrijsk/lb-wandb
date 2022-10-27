# W&B hyperparameter tuning: https://docs.wandb.ai/guides/sweeps

import argparse
import numpy as np
import wandb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_size", type=int, default=500)
    parser.add_argument("--n_samples", type=int, default=200)
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()

    wandb.init(project="LB1", entity="dmitrijs-modulai", config=args)
    config = wandb.config

    running_mean = 0
    for iter in range(1, config["n_samples"] + 1):

        sample = np.random.randn(config["sample_size"])
        sample_mean = np.mean(sample)
        running_mean += 1 / iter * (sample_mean - running_mean)

        wandb.log({"bias": np.abs(running_mean)})
