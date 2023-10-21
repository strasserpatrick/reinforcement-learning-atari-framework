import argparse

import gymnasium as gym
import torch

from dqn import RegularDQN

from pathlib import Path

LOOP_MODE = True

root_dir = Path(__file__).parent.parent.parent


def test_live(checkpoint_dir: Path):
    checkpoint_path = root_dir / checkpoint_dir
    # Load the model
    model: RegularDQN = RegularDQN.load_from_checkpoint(
        checkpoint_path=checkpoint_path
    )

    model.eval()

    # Initialize environment
    env = gym.make(
        model.hparams.env, render_mode="human"
    )

    while True:
        # Reset environment and get initial observation
        state, _ = env.reset()

        done = False

        while not done:
            # Convert state to tensor, add batch and channel dimensions
            state_tensor = (
                torch.tensor(state, dtype=torch.float32)
            )
            batched_state_tensor = state_tensor.unsqueeze(0)

            # Forward propagate the observation through the model
            q_values = model(batched_state_tensor)

            # Get the action that has the maximum Q-value
            action = torch.argmax(q_values).item()

            # Perform action and get new state and reward
            state, reward, done, truncated, info = env.step(action)

            env.render()

        if not LOOP_MODE:
            env.close()
            break
