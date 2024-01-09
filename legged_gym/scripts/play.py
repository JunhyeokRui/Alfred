# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym 
from isaacgym import gymtorch
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import csv
import datetime

import numpy as np
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.ext_force_robots = False
    env_cfg.domain_rand.ext_force_vector_6d = [-20, 0, 0.0, 0, 0, 0]
    env_cfg.domain_rand.ext_force_start_time = 3.0
    env_cfg.domain_rand.ext_force_duration = 0.2

    # commands
    env_cfg.commands.ranges.lin_vel_x = [1, 1]
    env_cfg.commands.ranges.lin_vel_y = [0,0] 
    env_cfg.commands.ranges.ang_vel_yaw = [0, 0]
    env_cfg.commands.ranges.heading = [0, 0]
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)
    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 700 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    # Rui
    dtt = datetime.datetime.now()
    dtTemp = dtt.strftime("%Y-%m-%d %H:%M")
    f3 = open('/home/dyros/test_ws/src/legged_gym/logs/sim2%s.csv' % dtTemp, 'w')
    writer3 = csv.writer(f3)    

    total_power = 0
    total_power_positive = 0
    total_base_lin_vel = 0
    

    for i in range(10*int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)
        current_time = i * env.dt
        #print(f'Time = {current_time:.2f}')
        
        action_scale = 1
        # Robot 0
        dof_pos=obs[0,15:21]  
        power_robot0=torch.sum(actions[0,:] * action_scale * obs[0, 21:27])
        q_init = torch.tensor( [0, 0, 0, 0 , 0, 0], device=device)
        k_joint = torch.tensor([0, 0, 0, 0 , 0, 0], device=device)
            
        spring_force_robot0 = -(k_joint*(dof_pos- q_init))
        spring_power_robot0 = torch.sum(spring_force_robot0  * obs[0, 21:27])
        contact_x =obs[0,0:1].item()
        contact_y = obs[0,1:2].item()

        for j in range(env_cfg.env.num_envs):

            dof_vel = obs[j, 21:27]
            power = torch.sum(actions[j,:] * action_scale * dof_vel)



        if 3 <= current_time <= 7:
            cotf_values=[]
            cotf_positive_values =[]

            for j in range(env_cfg.env.num_envs):
                
                dof_vel = obs[j, 21:27]
                base_lin_vel_x = obs[j, 3].item()
                #base_lin_vel_y = obs[j, 4].item()
                base_lin_vel = base_lin_vel_x
            
                power = torch.sum(actions[j,:] * action_scale * dof_vel)

                total_power += power

                power2 = torch.clamp(power, min=0)
                total_power_positive += power2

                total_base_lin_vel += base_lin_vel
                m= 1.2540778741240501+1
                cotf = (total_power) / ((m)* 9.80665  * total_base_lin_vel)
                cotf = cotf.item()
                cotf_positive = (total_power_positive) / ((m)* 9.80665 * total_base_lin_vel)
                cotf_positive = cotf_positive.item()

                cotf_values.append(cotf)
                cotf_positive_values.append(cotf_positive)
                if cotf<0:
                    print(f'Robot {j} COTF = {cotf:.10f}')

        if current_time > 7:
            average_cotf = sum(cotf_values) / len(cotf_values)
            average_cotf_positive = sum(cotf_positive_values) / len(cotf_positive_values)
            print(f'Average COTF = {average_cotf:.10f}')
            print(f'Average COTF_Positive = {average_cotf_positive:.10f}')
            #break

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),
            
                }
            )
            # Rui: save dof torques, dof acc
            logsim = [
                #angle
                env.dof_pos[0, 0].item(),
                env.dof_pos[0, 1].item(),
                env.dof_pos[0, 2].item(),
                env.dof_pos[0, 3].item(),
                env.dof_pos[0, 4].item(),
                env.dof_pos[0, 5].item(),
                #vel
                env.dof_vel[0, 0].item(),
                env.dof_vel[0, 1].item(),
                env.dof_vel[0, 2].item(),
                env.dof_vel[0, 3].item(),
                env.dof_vel[0, 4].item(),
                env.dof_vel[0, 5].item(),
                #env.dof_vel[0, 6].item(),
                #env.dof_vel[0, 7].item(),
                env.torques[0, 0].item(),
                env.torques[0, 1].item(),
                env.torques[0, 2].item(), ### kneee (check)
                env.torques[0, 3].item(),
                env.torques[0, 4].item(),
                env.torques[0, 5].item(),  ### knee
                #env.torques[0, 6].item(),
                #env.torques[0, 7].item(),
                env.commands[0, 0].item(), # x command
                env.commands[0, 1].item(), # y command
                env.commands[0, 2].item(), # angular command

                env.base_lin_vel[0, 0].item(), # x
                env.base_lin_vel[0, 1].item(), # y
                env.base_ang_vel[0, 2].item(), # z

                power_robot0.item(),  
                spring_power_robot0.item(), 
                contact_x, 
                contact_y,
                spring_force_robot0[0].item(),
                spring_force_robot0[1].item(),
                spring_force_robot0[2].item(),
                spring_force_robot0[3].item(),
                spring_force_robot0[4].item(),
                spring_force_robot0[5].item(),
                
                ]
            
            writer3.writerow(logsim)
        elif i==stop_state_log:
            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()
    # Rui
    f3.close()

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
