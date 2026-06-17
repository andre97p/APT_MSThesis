from marl_environment import PenGymMultiEnv
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.utils.typing import MultiAgentDict
from ray.rllib.policy.policy import PolicySpec
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from gymnasium import Space


"""
This file implements the Multi-Agent algorithms to train the agents in the environment
"""


temp_env= PenGymMultiEnv({})
obs_space_attacker = temp_env.observation_Space["attacker"]
act_space_attacker = temp_env.action_Space["attacker"]
obs_space_defender = temp_env.observation_Space["defender"]
act_space_defender = temp_env.action_Space["defender"]
GLOBAL_STATE_DIM = obs_space_attacker.shape[0] + obs_space_defender.shape[0]



class MappoArchitecture(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Extract spatial dimensions from the composite Dictionary space
        local_dim = 0
        global_dim = 0

        # 1. Decentralized Actor Network (\pi_{\theta})
        self.actor = nn.Sequential(
            nn.Linear(local_dim, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
            nn.Linear(256, num_outputs) # Outputs logits for the action distribution
        )

        # 2. Centralized Critic Network (V_{\phi})
        self.critic = nn.Sequential(
            nn.Linear(global_dim, 512),
            nn.LayerNorm(512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
            nn.Linear(256, 1) # Outputs standard scalar baseline estimate
        )
        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        # Unpack the composite dictionary injected by RLlib's sample batch
        local_obs = 1
        global_state = 1

        self._value_out = self.critic(global_state).squeeze(-1)

        action_logits = self.actor(local_obs)
        return action_logits, state

    def value_function(self):
        return self._value_out
    

def config_MAPPO():

    ModelCatalog.register_custom_model("mappo_model", MappoArchitecture)
    ray.init(ignore_reinit_error=True)
    tune.register_env("PenGymMultiEnv-v0", lambda config: PenGymMultiEnv(config))

    config = (
            PPOConfig()
            .environment("PenGymMultiEnv-v0")
            .framework("torch") # Use PyTorch
            .multi_agent(
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
                    )
                },
                # Map the agent string ID from the environment to the specific policy name
                policy_mapping_fn = lambda agent_id, episode: 
                    "attacker_policy" if agent_id == "attacker" else "defender_policy",
                )
            .training(#hyper parameters should be tuned
                lr=3e-4,
                clip_param=0.2,
                gamma=0.99,
                lambda_=0.95,  
                use_gae=True,
                train_batch_size=4000,
                sgd_minibatch_size=128,
                num_sgd_iter=10,
                model= {
                    "custom_model": "mappo_model",
                    "custom_model_config":{},
                })
            .rollouts(num_rollout_workers=2)
    )
    

def config_IPPO():
    ray.init(ignore_reinit_error=True)
    tune.register_env("PenGymMultiEnv-v0", lambda config: PenGymMultiEnv(config))

    config = (
        PPOConfig()
        .environment("PenGymMultiEnv-v0")
        .framework("torch") # Use PyTorch
        .multi_agent(
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
                )
            },
            # Map the agent string ID from the environment to the specific policy name
            policy_mapping_fn = lambda agent_id, episode: 
                "attacker_policy" if agent_id == "attacker" else "defender_policy",
            )
        .training(#hyper parameters should be tuned
            lr=3e-4,
            clip_param=0.2,
            gamma=0.99,
            lambda_=0.95,  
            use_gae=True,
            train_batch_size=4000,
            sgd_minibatch_size=128,
            num_sgd_iter=10)
        .rollouts(num_rollout_workers=1) # distributed training
        )
    return config
    

def execute_training(self, type):
    if type=="mappo":
        algo = self.config_IPPO().build()
    elif type=="ippo":
        algo = self.config_MAPPO().build()

    while(True):
        pass