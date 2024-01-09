from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
from legged_gym.envs import LeggedRobot

class Bolt(LeggedRobot):
    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        single_contact = torch.sum(1.*contacts, dim=1)==1
        return 1.*single_contact

    def _reward_energy(self):
        #vel*torque, value  scale or sum 
        energy=(self.torques*self.dof_vel)
        positive_energy=(self.torques*self.dof_vel).clip(min=0.)
        if self.cfg.rewards.positive_energy_reward:
            return torch.sum(torch.square(positive_energy), dim=1)
        else:
            return torch.sum(torch.square(energy), dim=1)
        ### sum over all columns of the square values for energy (using joint velocities)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        
        elif control_type=="S":
            q_init= torch.tensor([0.01,0.23, -0.45 , 0.01 , 0.25 , -0.39], device=self.device)
            k_joint = torch.tensor([7.48, -2.21, 1.64, 1.60, -1.26, 0.53], device=self.device)


            spring_torque = (k_joint*(actions - q_init))

            torques = actions_scaled + (spring_torque)


        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)


