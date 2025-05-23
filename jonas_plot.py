#!/usr/bin/env python3
#this code was generated by github copilot and proof-read and edited by Jonas Korkosh

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse


def read_progress_csv(filepath):
    """
    Reads a single progress.csv file, returning a list of mean batch rewards.
    Skips any lines starting with '#', which contain metadata.
    """
    df = pd.read_csv(filepath, comment="#")
    rewards = df["ep_reward_mean"].tolist()
    return rewards

def collect_all_progress_files(directory="."):
    """
    Collects all progress.csv files in the given directory and its subdirectories recursively.
    Returns a list of file paths.
    """
    pattern = os.path.join(directory, "**", "*progress.csv")
    return glob.glob(pattern, recursive=True)

def compute_reward_statistics(reward_arrays):
    """
    Given a list of lists (reward_arrays), where each list contains per-batch reward data
    from one run, compute arrays for:
      - batch indices (x-axis)
      - mean reward
      - standard deviation

    Returns: (batches, means, stds)
    """
    max_len = max(len(arr) for arr in reward_arrays)
    means = []
    stds = []

    for batch_idx in range(max_len):
        batch_rewards = []
        for arr in reward_arrays:
            if batch_idx < len(arr):
                batch_rewards.append(arr[batch_idx])
        if batch_rewards:
            means.append(np.mean(batch_rewards))
            stds.append(np.std(batch_rewards))
        else:
            break

    batches = np.arange(1, len(means) + 1)
    return batches, means, stds

def plot_multiple_configs(dir_label_batches_pairs, append=False, title="Rewards vs. Batches"):
    """
    For each (directory, label) pair:
      1) Collect all progress.csv files
      2) Read the reward data
      3) Compute batches/means/stds
      4) Plot them with a given label
    All on the same figure.
    """
    plt.figure(figsize=(8, 5))
    last_x = 0
    for directory, label, num_batches in dir_label_batches_pairs:
        file_paths = collect_all_progress_files(directory)
        if not file_paths:
            print(f"No progress.csv files found in '{directory}', skipping.")
            continue

        # Gather all runs in this directory into a list of reward arrays
        reward_runs = [read_progress_csv(fp) for fp in file_paths]

        # Compute the statistics
        batches, means, stds = compute_reward_statistics(reward_runs)
        if num_batches:
            batches = batches[:num_batches]
            means = means[:num_batches]
            stds = stds[:num_batches]
        if append:
            batches = np.array(batches) + last_x
        # Plot on the same figure
        plt.plot(batches, means, label=label)
        plt.fill_between(
            batches,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            alpha=0.2
        )

        last_x = batches[-1]

    plt.title(title)
    plt.xlabel("Batch Index")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def generate_default_label(dirpath):
    """
    Attempt to parse a directory path of the form:
        /.../logs/hardware/QubeSwingupEnv/1e6/seed-984
    Returning a label like:
        hardware/QubeSwingupEnv/1e6/seed-984
    Falls back to "UnknownRun" if parsing fails.
    """
    label = "UnknownRun"
    parts = dirpath.split("/")
    try:
        logs_idx = parts.index("logs")
        # everything that comes after "logs" is joined into the label
        label = "/".join(parts[logs_idx + 1 :])
    except (ValueError, IndexError):
        pass
    return label

def main():
    parser = argparse.ArgumentParser(description="Plot rewards from monitor.csv files.")
    parser.add_argument(
        "-d",
        "--directories",
        type=str,
        nargs="+",
        default=["."],
        help="List of directories in which to search for progress.csv files."
    )
    parser.add_argument(
        "-l",
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help="List of labels corresponding to each directory."
    )
    parser.add_argument(
        "-nb",
        "--num-batches",
        type=int,
        nargs="+",
        default=None,
        help="Number of batches to plot for each corresponding directory. If 0, all batches will be plotted."
    )
    parser.add_argument(
        "-a",
        "--append",
        action="store_true",
        default=False,
        help="Append to the end of the existing plot instead of starting at x=0. Runs in the order specified in the directories argument."
    )
    parser.add_argument(
        "-t",
        "--title",
        type=str,
        default="Rewards vs. Batches",
        help="Title of the plot."
    )
    args = parser.parse_args()

    if args.labels is None or len(args.labels) != len(args.directories):
        # Create default labels if not provided or mismatch in length
        labels = [generate_default_label(dirpath) for dirpath in args.directories]
    else:
        labels = args.labels
    if args.num_batches is None or len(args.num_batches) != len(args.directories):
        # Create default num_batches if not provided or mismatch in length
        print("Warning: num_batches length does not match number of directories, setting all to 0")
        num_batches = [0 for _ in range(len(args.directories))]
    else:
        num_batches = args.num_batches

    dir_label_batches_pairs = list(zip(args.directories, labels, num_batches))
    plot_multiple_configs(dir_label_batches_pairs, append=args.append, title=args.title)

if __name__ == "__main__":
    main()