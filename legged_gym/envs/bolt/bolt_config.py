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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class BoltCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env):
        num_envs = 4096 # robot count
        # num_observations = 151 # not sure
        num_observations = 30
        num_actions = 6 # robot actuation
        
    
    class terrain( LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        measure_heights = False
        #vertical_scale = 0.001 # [m] Rui
        
        #curriculum = False # Rui
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete] # Rui
        #terrain_proportions = [0.0, 1.0, 0.0, 0.0, 0.0] # Rui
        # trimesh only:
        #slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces # Rui

    
    class command( LeggedRobotCfg.commands):
        num_commands = 3
        heading_command = False
        class ranges:
            lin_vel_x = [0.3, 0.3] 
            lin_vel_y = [0., 0.] 
            ang_vel_yaw = [0., 0.]    # min max [rad/s] ## refer to heading_command function
            # lin_vel_x = [0.3, 0.3] 
            # lin_vel_y = [0.3, 0.3] 
            # ang_vel_yaw = [0.3, 0.3]    # min max [rad/s] ## refer to heading_command function
    
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.4] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_HAA': 0.,
            # 'hip_rotation_left': 0.,
            'FL_HFE': 0.1,
            # 'thigh_joint_left': -1.8,
            'FL_KFE': -0.25,
            'FL_ANKLE': 0.,

            'FR_HAA': 0.,
            # 'hip_rotation_right': 0.,
            'FR_HFE': 0.1,
            # 'thigh_joint_right': -1.8,
            'FR_KFE': -0.25,
            'FR_ANKLE': 0.
        }

    class control( LeggedRobotCfg.control ):
        
        control_type = 'S' # P: position, V: velocity, T: torques,  S: Spring (refer to bolt.py)
        
        # PD Drive parameters:
        stiffness = {   'FL_HAA': 0.1, 
                        # 'hip_rotation': 100.0,
                        'FL_HFE': 0.2,
                        'FL_KFE': 0.2,
                        
                        # 'thigh_joint': 200.,
                        'FR_HAA': 0.1,
                        'FR_HFE': 0.2,
                        'FR_KFE': 0.2,
                        
                        #'FL_ANKLE': 200.,
                        #'FR_ANKLE': 200.
                        # 'toe_joint': 40.
                        }  # [N*m/rad]
        damping = { 'FL_HAA': 0.02, 
                    'FL_HFE': 0.02,
                    'FL_KFE': 0.01,
                    'FR_HAA': 0.02, 
                    
                    'FR_HFE': 0.02,
                    
                    'FR_KFE': 0.01,
                    
                    #'FL_ANKLE': 6.,
                    #'FR_ANKLE': 6.
                   #'hip_rotation': 3.0,
                   #'hip_flexion': 6., 
                    # 'thigh_joint': 6., 
                    # 'ankle_joint': 6.,
                    # 'toe_joint': 1.
                    }  # [N*m*s/rad]     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5 # 0.5 in pos control, was 1(Alfred)
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 1
        
        
    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/bolt/urdf/bolt.urdf'
        name = "bolt"
        foot_name = 'FR_FOOT'
        # penalize_contacts_on = ['bolt_lower_leg_right_side', 'bolt_body', 'bolt_hip_fe_left_side', 'bolt_hip_fe_right_side', ' bolt_lower_leg_left_side', 'bolt_shoulder_fe_left_side', 'bolt_shoulder_fe_right_side', 'bolt_trunk', 'bolt_upper_leg_left_side', 'bolt_upper_leg_right_side']
        # penalize_contacts_on = ['base_link', 'FR_SHOULDER', 'FL_SHOULDER', 'FR_LOWER_LEG', 'FL_LOWER_LEG']
        terminate_after_contacts_on = ['base_link', 'FR_SHOULDER', 'FL_SHOULDER']
        disable_gravity = False
        flip_visual_attachments = False
        # fix_base_link = True
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        max_angular_velocity = 5.
        max_linear_velocity = 5. #100 default

    class rewards:
        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma), dont forget to change to nonzero values
        soft_dof_pos_limit = 0.95 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.95
        soft_torque_limit = 0.95
        base_height_target = 0.
        max_contact_force = 0. # forces above this value are penalized
        positive_energy_reward = False

        class scales:
            termination = -50.0 ## Tuned. 
            tracking_lin_vel = 10. ## Tuned. more if static
            tracking_ang_vel= 1. ## RETUNED: Increase for positive energy to see a better symmetry then tune energy again 

            dof_pos_limits = -1. #Standard
            torque_limits= -0.0001 # Tuned. if static change to zero (knee torques show 2Nm constant)
            dof_vel_limits= -1. # Standard

            energy= -0.000005 # Tuned. increase if robot moves energetic
                       
            lin_vel_z = -0.
            ang_vel_xy = -0.
            orientation = -0.

            torques = 0.
            dof_vel = -0.
            dof_acc = -0       

            base_height = -0. 
            feet_air_time =  0.
            collision = -0. 
            feet_stumble = -0.
            action_rate = -0.01 # Standard. not tuned yet (?)
            #Makes torques less noisy, better vel tracking. 
            stand_still = -0.

class BoltCfgPPO( LeggedRobotCfgPPO ):
    
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = 'bolt_test'
        experiment_name = 'bolt_test'

    class algorithm( LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01