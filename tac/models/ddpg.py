import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tac.models.model import TrajectoryModel
class DDPGActor(TrajectoryModel):
    def __init__(self, state_dim, act_dim, max_length=None, max_action=1.0,**kwargs):
        super().__init__(state_dim, act_dim, max_length)
        self.l1 = nn.Linear(state_dim, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, act_dim)

        self.max_action = max_action

    def forward(self, states, actions, rewards, timestep,masks=None, attention_mask=None):
        a = F.relu(self.l1(states))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

    def get_action(self, states, actions, rewards, timesteps, **kwargs):
        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.act_dim)
        rewards = rewards.reshape(-1, 1)
        timesteps = timesteps.reshape(1, -1)
        last_state = states[-1]
        states = states.to(dtype=torch.float32)
        return self.forward(states, actions, rewards, timesteps, **kwargs)[0]