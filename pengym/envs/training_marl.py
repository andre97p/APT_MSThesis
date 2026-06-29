from pengym.envs.marl_environment import PenGymMultiEnv
from pengym.envs.plot_results import plot_rewards, plot_steps
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.utils.typing import MultiAgentDict
from ray.rllib.policy.policy import PolicySpec
import torch
import torch.nn as nn
from torch import zeros
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from gymnasium.spaces.utils import flatdim
import numpy as np


"""
This file implements the Multi-Agent algorithms to train the agents in the environment
"""


temp_env= PenGymMultiEnv(scenario_name="tiny")
obs_space_attacker = temp_env.observation_Space["attacker"]
act_space_attacker = temp_env.action_Space["attacker"]
obs_space_defender = temp_env.observation_Space["defender"]
act_space_defender = temp_env.action_Space["defender"]


LOCAL_DIM_ATT= flatdim(obs_space_attacker)
LOCAL_DIM_DEF = flatdim(obs_space_defender)
GLOBAL_STATE_DIM = LOCAL_DIM_ATT + LOCAL_DIM_DEF  #we assume the conjunction of these sets represents the entire PenGym state


class MappoArchitecture(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Extract spatial dimensions from the composite Dictionary space
        local_dim = flatdim(obs_space)
        global_dim = model_config.get("custom_model_config", {}).get("global_dim", GLOBAL_STATE_DIM)
        self._global_dim = int(global_dim)
        # 1. Decentralized Actor Network (\pi_{\theta})
        self.actor = nn.Sequential(
            nn.Linear(local_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, num_outputs) # Outputs logits for the categorical action distribution
        )

        # 2. Centralized Critic Network (V_{\phi})
        self.critic = nn.Sequential(
            nn.Linear(global_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.1),
            nn.Linear(512,256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 1) #Outputs standard scalar baseline estimate
        )
        self._value_out = None

    def _flatten_tensors(self, obs_tensor):
        """
        Recursively flattens nested dictionaries of tensors into a single 2D representation.
        Gymnasium strict requirement: dictionary keys MUST be sorted alphabetically.
        """
        
        if isinstance(obs_tensor, dict):
            flat_list = []
            # We must maintain alphabetical order to align with the Neural Network input layer
            for key in sorted(obs_tensor.keys()):
                flat_list.append(self._flatten_tensors(obs_tensor[key]))
            return torch.cat(flat_list, dim=-1)
        else:
            # Guarantee 2D shape [BatchSize, Features] to prevent torch.cat dimension mismatches
            return obs_tensor.view(obs_tensor.shape[0], -1)

    def forward(self, input_dict, state, seq_lens):

        if isinstance(input_dict["obs"], dict) and "local_obs" in input_dict["obs"]:
            local_obs_raw = input_dict["obs"]["local_obs"]
        else:
            local_obs_raw = input_dict["obs"]
            
        local_obs = self._flatten_tensors(local_obs_raw)

        if isinstance(input_dict["obs"], dict) and "global_state" in input_dict["obs"]:
            global_state = self._flatten_tensors(input_dict["obs"]["global_state"])
        else:
            # Bypass the critic crash by feeding a zero tensor mapped to the correct device
            global_state = torch.zeros(
                size=(local_obs.shape[0], self._global_dim), 
                dtype=torch.float32, 
                device=local_obs.device
            )

        # Critic forward pass (Omniscient Baseline)
        self._value_out = self.critic(global_state).squeeze(-1)

        # Actor forward pass (Partial Observability Policy)
        action_logits = self.actor(local_obs)
        
        return action_logits, state
    
    def value_function(self):
        return self._value_out
    

def config_MAPPO()->PPOConfig:

    ModelCatalog.register_custom_model("mappo_model", MappoArchitecture)
    ray.init(ignore_reinit_error=True)
    tune.register_env("PenGymMultiEnv-v0", lambda env_config: PenGymMultiEnv(scenario_name="tiny"))

    config = PPOConfig()

    config.environment("PenGymMultiEnv-v0", env_config={
        "scenario_name": "tiny",
        "enable_nasim": True,
        "enable_pengym": False,
    })
    config.framework("torch") # Use PyTorch
    config.api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False
    )
    config.multi_agent(
        policies= {
            "attacker_policy": PolicySpec(
                observation_space=obs_space_attacker,
                action_space=act_space_attacker,
                config={}
            ),
            "defender_policy": PolicySpec(
                observation_space=obs_space_defender,
                action_space=act_space_defender,
                config={}
            )},
        policy_mapping_fn = lambda agent_id, episode, **kwargs:
            "attacker_policy" if agent_id == "attacker" else "defender_policy")
        # Map the agent string ID from the environment to the specific policy name
    config.training(
        #hyper parameters should be tuned
        lr=1e-4,
        clip_param=0.2,
        gamma=0.99,
        lambda_=0.95,
        use_gae=True,
        model= {
            "custom_model": "mappo_model",
            "custom_model_config": {
                "global_dim": GLOBAL_STATE_DIM
            }
        })
    config.env_runners(num_env_runners=1)
    return config
    

def config_IPPO()->PPOConfig:
    ray.init(ignore_reinit_error=True)
    tune.register_env("PenGymMultiEnv-v0", lambda env_config: PenGymMultiEnv(scenario_name=env_config.get("scenario_name", "tiny")))

    config = PPOConfig()
    config.api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False
    )
    config.environment("PenGymMultiEnv-v0", env_config={
        "scenario_name": "tiny",
        "enable_nasim": True,
        "enable_pengym": False,
    })
    config.framework("torch") # Use PyTorch
    config.multi_agent(
        policies= {
            "attacker_policy": PolicySpec(
                observation_space=obs_space_attacker,
                action_space=act_space_attacker,
                config={}
            ),
            "defender_policy": PolicySpec(
                observation_space=obs_space_defender,
                action_space=act_space_defender,
                config={}
            )},
        policy_mapping_fn = lambda agent_id, episode, **kwargs:
            "attacker_policy" if agent_id == "attacker" else "defender_policy")
        # Map the agent string ID from the environment to the specific policy name
                        
    config.training(
        lr=1e-4,
        clip_param=0.2,
        gamma=0.99,
        lambda_=0.95,  
        use_gae=True,
        num_sgd_iter=10,)
    config.env_runners(num_env_runners=1)

    return config
    

def execute_training(algo_type, training_iterations=4000):
    """
    Initializes and executes the MARL training loop using RLlib.

    Returns:
        history (dict) with keys:
            'attacker_reward'  - mean reward for attacker_policy per iteration
            'defender_reward'  - mean reward for defender_policy per iteration
            'episode_len_mean' - mean steps per episode per iteration
    """
    history = {
        'attacker_reward':  [],
        'defender_reward':  [],
        'episode_len_mean': [],
    }

    config: PPOConfig
    if algo_type == "mappo":
        print("[*] Compiling MAPPO Configuration (CTDE)...")
        config = config_MAPPO()
    elif algo_type == "ippo":
        print("[*] Compiling IPPO Configuration (Decentralized)...")
        config = config_IPPO()
    else:
        raise ValueError("Algorithm type must be 'mappo' or 'ippo'")

    print(f"[*] Building {algo_type.upper()} algorithm graph...")
    algo = config.build_algo()

    print(f"[*] Starting {algo_type.upper()} training loop for {training_iterations} iterations...")

    for i in range(training_iterations):
        result = algo.train()
        policy_rewards = result["env_runners"]['policy_reward_mean']
        att_rew = policy_rewards['attacker_policy']
        def_rew = policy_rewards['defender_policy']
        ep_len  = int(result['env_runners']['episode_len_mean'])

        history['attacker_reward'].append(att_rew)
        history['defender_reward'].append(def_rew)
        history['episode_len_mean'].append(ep_len)

        print(f"Iteration {i+1}/{training_iterations} | "
              f"Attacker Return: {att_rew} | "
              f"Defender Return: {def_rew} | "
              f"Avg Steps: {ep_len}")
        
        """if (i + 1) % 100 == 0:
            checkpoint_dir = algo.save(checkpoint_dir=f"./checkpoints/{algo_type}_{i+1}")
            print(f"[*] Checkpoint saved at: {checkpoint_dir}")
        """
    plot_rewards(history, algo_type=algo_type)
    plot_steps(history, algo_type=algo_type)

    return history
