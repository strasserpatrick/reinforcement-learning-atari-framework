from pathlib import Path

import numpy as np
import pandas as pd

root_directory = Path(__file__).parent.parent


def load_dicts(path):
    df = pd.read_csv(path)
    df = df[["validation_reward_epoch", "step"]]

    # filter out nan rows
    df = df[df["validation_reward_epoch"].notna()]

    result_dict = {key: value for key, value in zip(df['step'], df['validation_reward_epoch'])}
    return result_dict


def join_dicts(*dicts):
    res_dict = {}
    for dict in dicts:
        for key, value in dict.items():
            if key in res_dict:
                res_dict[key].append(value)
            else:
                res_dict[key] = [value]
    return res_dict


def get_mean_dict(dict):
    res_dict = {}
    for key, value in dict.items():
        res_dict[key] = np.mean(value)
    return res_dict


def get_std_dict(dict):
    res_dict = {}
    for key, value in dict.items():
        res_dict[key] = np.std(value)
    return res_dict

def add_plot_line(df, ax, label, color):
    ax.plot(df["steps"], df["mean"], color=color, label=label)
    ax.fill_between(df["steps"], df["mean"] - df["std"], df["mean"] + df["std"], color=color, alpha=0.08)


def get_multiple_runs_dict(base_path: Path):
    base_path = root_directory / base_path
    path_numbers = [path.name for path in base_path.iterdir() if path.is_dir()]

    dicts = [load_dicts(str(root_directory / base_path / str(number) / "metrics.csv")) for number in path_numbers]
    data_dict = join_dicts(*dicts)

    df = pd.DataFrame({'mean': pd.Series(get_mean_dict(data_dict)),
                       'std': pd.Series(get_std_dict(data_dict))})

    df.index.name = "steps"
    df.reset_index(inplace=True)

    return df
