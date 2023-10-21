import sys
from pathlib import Path
from typing import Annotated

import typer

root_directory = Path(__file__).parent.parent
sys.path.append(str(root_directory / "src"))

from utils.config import DQNType, Networks, ReplayBufferTypes, DQNConfig, parse_yaml_from_file, Device, \
    DefaultValues
from utils.dqn_test_live import test_live
from utils.dqn_test_minatar_live import MinAtarLiveTest
from utils.checkpoint_finder import get_newest_checkpoint_path
from dqn import dqn_factory
from utils.dqn_train import train_dqn

app = typer.Typer()


@app.command()
def train(
        file_name: str = typer.Option("", "-f", help="name of the file to save the model"),
        max_epochs: int = typer.Option(DefaultValues.max_epochs, "--max_epochs",
                                       help="max number of epochs to train for"),
        dqn_type: DQNType = typer.Option(DefaultValues.dqn_type, "--dqn_type", help="type of DQN"),
        batch_size: int = typer.Option(DefaultValues.batch_size, "--batch_size", help="size of the batches"),
        lr: float = typer.Option(DefaultValues.lr, "--lr", help="learning rate"),
        network_type: Networks = typer.Option(DefaultValues.network_type, "--network_type",
                                              help="neural network architecture"),
        replay_buffer_type: ReplayBufferTypes = typer.Option(DefaultValues.replay_buffer_type, "--replay_buffer_type",
                                                             help="type of replay buffer"),
        hidden_size: int = typer.Option(DefaultValues.hidden_size, "--hidden_size", help="hidden layer size"),
        num_hidden_layers: int = typer.Option(DefaultValues.num_hidden_layers, "--num_hidden_layers",
                                              help="number of hidden layers. Note: this plays only a role when selecting MLP as network architecture"),
        env: str = typer.Option(DefaultValues.env, "--env", help="gym environment tag"),
        gamma: float = typer.Option(DefaultValues.gamma, "--gamma", help="discount factor"),
        sync_rate: int = typer.Option(DefaultValues.sync_rate, "--sync_rate",
                                      help="how many frames do we update the target network"),
        replay_size: int = typer.Option(DefaultValues.replay_size, "--replay_size",
                                        help="capacity of the replay buffer"),
        warm_start_steps: int = typer.Option(DefaultValues.warm_start_steps, "--warm_start_steps",
                                             help="transitions to fill buffer in the beginning"),
        eps_start: float = typer.Option(DefaultValues.eps_start, "--eps_start", help="starting value of epsilon"),
        eps_end: float = typer.Option(DefaultValues.eps_end, "--eps_end", help="final value of epsilon"),
        eps_decay: int = typer.Option(DefaultValues.eps_decay, "--eps_decay", help="length of epsilon decay"),
        episode_length: int = typer.Option(DefaultValues.episode_length, "--episode_length",
                                           help="max length of an episode"),
        max_episode_reward: int = typer.Option(DefaultValues.max_episode_reward, "--max_episode_reward",
                                               help="max episode reward in the environment"),
        validate_every_n_epochs: int = typer.Option(DefaultValues.validate_every_n_epochs, "--validate_every_n_epochs",
                                                    help="how often to validate the model"),
        rollouts_per_validation: int = typer.Option(DefaultValues.rollouts_per_validation, "--rollouts_per_validation",
                                                    help="n rollouts per validation step"),
        device: Device = typer.Option(DefaultValues.device, "--device",
                                      help="device type, either 'cuda', 'mps' or 'cpu'")
):
    if file_name:
        print(f"Training model from file {file_name}")

        config = parse_yaml_from_file(str(root_directory / file_name))
        model = dqn_factory(config.dqn_type, hparams=config)

    else:
        config = DQNConfig(
            max_epochs=max_epochs,
            batch_size=batch_size,
            lr=lr,
            network_type=network_type,
            replay_buffer_type=replay_buffer_type,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            env=env,
            gamma=gamma,
            sync_rate=sync_rate,
            replay_size=replay_size,
            warm_start_steps=warm_start_steps,
            eps_start=eps_start,
            eps_end=eps_end,
            eps_decay=eps_decay,
            episode_length=episode_length,
            max_episode_reward=max_episode_reward,
            validate_every_n_epochs=validate_every_n_epochs,
            rollouts_per_validation=rollouts_per_validation,
            device=device
        )
        model = dqn_factory(dqn_type, hparams=config)
    train_dqn(model=model, config=config)


@app.command()
def minatar_live(checkpoint_path: Annotated[
    Path, typer.Option("--checkpoint_path", help="Path to checkpoint file", prompt="Specify the checkpoint path")]):
    if str(checkpoint_path) == "newest":
        checkpoint_path = get_newest_checkpoint_path()
    MinAtarLiveTest(checkpoint_path)


@app.command()
def live(checkpoint_path: Annotated[
    Path, typer.Option("--checkpoint_path", help="Path to checkpoint file", prompt="Specify the checkpoint path")]):
    if str(checkpoint_path == "newest"):
        checkpoint_path = get_newest_checkpoint_path()
    test_live(checkpoint_path)


if __name__ == "__main__":
    app()
