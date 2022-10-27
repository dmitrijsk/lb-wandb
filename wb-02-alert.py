# W&B alerts: https://docs.wandb.ai/guides/track/alert

import numpy as np
import wandb

config = {"sample_size": 500, "n_samples": 200, "eps": 0.01}

wandb.init(project="LB2", entity="dmitrijs-modulai", config=config)

running_mean = 0
samples = []
means = []
for iter in range(1, config["n_samples"] + 1):

    sample = np.random.randn(config["sample_size"])
    sample_mean = np.mean(sample)
    running_mean += 1 / iter * (sample_mean - running_mean)

    if np.abs(running_mean) < config["eps"]:
        wandb.alert(
            title="Converged",
            text=f"Running mean {running_mean} within {config['eps']}",
        )

    wandb.log({"running_mean": running_mean, "sample": sample})
