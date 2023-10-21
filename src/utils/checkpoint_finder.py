import glob
import os
from pathlib import Path

from dqn import AbstractDQN, dqn_model_finder
from utils.config import DQNType

project_root = Path(__file__).parent.parent.parent


def get_newest_checkpoint_path():
    # /**/ searches subdirectories recursively
    # /*.ckpt => finds all checkpoints
    file_list = glob.glob(f"logs/**/*.ckpt", recursive=True)

    if not file_list:
        return None

    # Get the newest file according to ctime (creation time on Windows)
    newest_file = max(file_list, key=os.path.getctime)
    print(f"found newest checkpoint: {newest_file}")

    return newest_file


def find_checkpoint_model(checkpoint_path: Path) -> AbstractDQN:
    hparams_yaml = checkpoint_path.parent.parent / "hparams.yaml"
    hparams_yaml = project_root / hparams_yaml
    if not hparams_yaml.exists():
        raise FileNotFoundError(f"Could not find hparams.yaml in {checkpoint_path.parent}")

    with open(hparams_yaml) as f:
        while True:
            line = f.readline()
            if not line:
                break

            if line.startswith("dqn_type: "):
                dqn_type = line.split(":")[1].strip()
                break

    return dqn_model_finder(DQNType(dqn_type))
