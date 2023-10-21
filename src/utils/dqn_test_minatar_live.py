from pathlib import Path

import gymnasium as gym
import torch
from minatar.gui import GUI

from utils.checkpoint_finder import find_checkpoint_model

LOOP_MODE = True

project_root = Path(__file__).parent.parent.parent


class MinAtarLiveTest:

    def __init__(self, checkpoint_dir: Path):
        # Load the model
        model = find_checkpoint_model(checkpoint_dir)

        checkpoint_path = project_root / checkpoint_dir
        self.model = model.load_from_checkpoint(
            checkpoint_path=checkpoint_path
        )

        self.my_device = torch.device(self.model.hparams.device)
        self.model = self.model.to(self.my_device)
        self.model.eval()

        env = self.model.hparams.env
        n_channels = 0

        if env == "MinAtar/SpaceInvaders":
            n_channels = 6
        elif env == "MinAtar/Freeway":
            n_channels = 7
        elif env == "MinAtar/Breakout-v1":
            n_channels = 4

        title = env.split("/")[1]

        # Initialize environment
        self.gui = GUI(title, n_channels)  # 4 color channels is a good choice!
        self.env = gym.make(self.model.hparams.env)

        self.is_terminated = False
        self.G = 0
        self.state = None
        self.initialize()

        # Initiate the game play
        self.gui.update(0, self.play)
        self.gui.run()

    def initialize(self):
        self.state, _ = self.env.reset()
        self.is_terminated = False
        self.G = 0

    def play(self):
        self.gui.display_state(self.state)

        if self.is_terminated:
            self.gui.set_message("Game over! Score: " + str(self.G))
            self.initialize()
            self.gui.update(1000, self.play)
            return

        self.gui.set_message("Score: " + str(self.G))

        state_tensor = (torch.tensor(self.state, dtype=torch.float32)).to(self.my_device)

        # Turns the tensor from size [10, 10, 4] to [1, 10, 10, 4] which is what the model expects
        # [batch_size, width, height, channels]
        state_tensor_with_batch_size = state_tensor.unsqueeze(0)
        q_values = self.model(state_tensor_with_batch_size)

        action = torch.argmax(q_values).item()

        self.state, reward, done, truncated, info = self.env.step(action)
        self.is_terminated = done or truncated

        self.G += reward

        self.gui.update(50, self.play)
