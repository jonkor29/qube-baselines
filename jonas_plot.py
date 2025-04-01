import pandas as pd

def read_progress_csv(filepath):
    """
    Reads a single progress.csv file, returning a list of mean batch rewards.
    Skips any lines starting with '#', which contain metadata.
    """
    df = pd.read_csv(filepath, comment="#")
    rewards = df["ep_reward_mean"].tolist()
    return rewards
