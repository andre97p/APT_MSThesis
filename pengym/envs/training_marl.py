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


class ActorCriticArchitecture(TorchModelV2, nn.Module):
    """
    Shared actor-critic network for both MARL algorithms.

    The actor is always decentralized (conditioned on the local observation only).
    The critic input is selected through custom_model_config["centralized_critic"]:
      - True  (MAPPO): critic estimates V(s) from the joint global state (CTDE)
      - False (IPPO):  critic estimates V(o_i) from the agent's local observation
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Extract spatial dimensions from the composite Dictionary space
        local_dim = flatdim(obs_space)
        custom_config = model_config.get("custom_model_config", {})
        self._centralized_critic = bool(custom_config.get("centralized_critic", True))
        self._global_dim = int(custom_config.get("global_dim", GLOBAL_STATE_DIM))

        # MAPPO conditions the baseline on the global state, IPPO on the local view
        critic_input_dim = self._global_dim if self._centralized_critic else local_dim

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

        # 2. Critic Network (V_{\phi}): centralized for MAPPO, decentralized for IPPO
        self.critic = nn.Sequential(
            nn.Linear(critic_input_dim, 512),
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

        if not self._centralized_critic:
            # IPPO: the value baseline only conditions on the agent's own observation
            critic_input = local_obs
        elif isinstance(input_dict["obs"], dict) and "global_state" in input_dict["obs"]:
            critic_input = self._flatten_tensors(input_dict["obs"]["global_state"])
        else:
            # Bypass the critic crash by feeding a zero tensor mapped to the correct device
            critic_input = torch.zeros(
                size=(local_obs.shape[0], self._global_dim),
                dtype=torch.float32,
                device=local_obs.device
            )

        # Critic forward pass (Omniscient Baseline for MAPPO, local baseline for IPPO)
        self._value_out = self.critic(critic_input).squeeze(-1)

        # Actor forward pass (Partial Observability Policy)
        action_logits = self.actor(local_obs)
        
        return action_logits, state
    
    def value_function(self):
        return self._value_out
    

def config_MAPPO()->PPOConfig:

    ModelCatalog.register_custom_model("actor_critic_model", ActorCriticArchitecture)
    ray.init(ignore_reinit_error=True)
    tune.register_env("PenGymMultiEnv-v0", lambda env_config: PenGymMultiEnv(scenario_name=env_config.get("scenario_name", "tiny"), env_config=env_config))

    config = PPOConfig()

    config.environment("PenGymMultiEnv-v0", env_config={
        "scenario_name": "tiny",
        "enable_nasim": False,
        "enable_pengym": True,
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
        lr=3e-4,
        clip_param=0.2,
        gamma=0.90,
        lambda_=0.95,
        use_gae=True,
        model= {
            "custom_model": "actor_critic_model",
            "custom_model_config": {
                "centralized_critic": True,
                "global_dim": GLOBAL_STATE_DIM
            }
        }
        )
    config.env_runners(num_env_runners=1)
    return config
    

def config_IPPO()->PPOConfig:

    ModelCatalog.register_custom_model("actor_critic_model", ActorCriticArchitecture)
    ray.init(ignore_reinit_error=True)
    tune.register_env("PenGymMultiEnv-v0", lambda env_config: PenGymMultiEnv(scenario_name=env_config.get("scenario_name", "tiny"), env_config=env_config))

    config = PPOConfig()
    config.api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False
    )
    config.environment("PenGymMultiEnv-v0", env_config={
        "scenario_name": "tiny",
        "enable_nasim": False,
        "enable_pengym": True,
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
        lr=3e-4,
        clip_param=0.15,
        gamma=0.90,
        lambda_=0.95,
        use_gae=True,
        model= {
            "custom_model": "actor_critic_model",
            "custom_model_config": {
                # Same network as MAPPO, but the critic stays decentralized:
                # it conditions on the agent's local observation, not the global state
                "centralized_critic": False
            }
        }
        )
    config.env_runners(num_env_runners=1)

    return config
    

def _run_training_iteration(algo, iteration, total_iterations, history):
    """
    Runs a single RLlib training iteration (one algo.train() call) and appends
    its metrics to `history`.

    """
    result = algo.train()
    env_runners = result.get("env_runners", {})
    policy_rewards = env_runners.get('policy_reward_mean', {})
    att_rew = policy_rewards.get('attacker_policy')
    def_rew = policy_rewards.get('defender_policy')
    ep_len_mean = env_runners.get('episode_len_mean')

    progress = f"Iteration {iteration}/{total_iterations}"

    # A policy key is only populated once at least one full episode for it
    # completed within this iteration's sample batch. It can be legitimately
    # missing when a RolloutWorker restarts mid-episode (e.g. after the
    # SYSTEM_ERROR crashes seen with the live cyber-range backend) or when no
    # episode finished yet. Skip logging that iteration instead of crashing
    # the whole run on a single flaky rollout.
    if att_rew is None or def_rew is None or ep_len_mean is None:
        print(f"{progress} | "
              f"No completed episodes reported for one or more policies "
              f"(worker likely restarted) - skipping this iteration's history.")
        return

    ep_len = int(ep_len_mean)

    history['iteration'].append(iteration)
    history['attacker_reward'].append(att_rew)
    history['defender_reward'].append(def_rew)
    history['episode_len_mean'].append(ep_len)

    print(f"{progress} | "
          f"Attacker Return: {att_rew} | "
          f"Defender Return: {def_rew} | "
          f"Avg Steps: {ep_len}")


def execute_training(algo_type, num_episodes, iterations_per_episode):
    """
    Initializes and executes the MARL training loop using RLlib.

    """
    history = {
        'iteration':        [],
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

    total_iterations = num_episodes * iterations_per_episode
    print(f"[*] Starting {algo_type.upper()} training loop for {total_iterations} iterations...")

    for iteration in range(1, total_iterations + 1):
        _run_training_iteration(algo, iteration, total_iterations, history)

        # Periodic checkpointing (disabled): save every `iterations_per_episode` iters.
        # if iteration % iterations_per_episode == 0:
        #     checkpoint_dir = algo.save(checkpoint_dir=f"./checkpoints/{algo_type}_iter{iteration}")
        #     print(f"[*] Checkpoint saved at: {checkpoint_dir}")

    plot_rewards(history, algo_type=algo_type)
    plot_steps(history, algo_type=algo_type)

    return history
