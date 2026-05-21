import gymnasium as gym
from gymnasium.utils import seeding
from gymnasium.spaces import Discrete
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers, AgentSelector
from environment import PenGymEnv
import pengym.utilities as utils
import functools
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig


def env(scenario, fully_obs=False, flat_actions=True, flat_obs=True):
    """
    The env function often wraps the environment in standard PettingZoo wrappers.
    """
    internal_env = PenGymMultiEnv(scenario, fully_obs, flat_actions, flat_obs)
    internal_env = wrappers.TerminateIllegalWrapper(internal_env, illegal_reward=-1)
    internal_env = wrappers.AssertOutOfBoundsWrapper(internal_env)
    internal_env = wrappers.OrderEnforcingWrapper(internal_env)
    return internal_env

class PenGymMultiEnv(AECEnv):

    """
    A PettingZoo extension of Pengym environment paradigm
    """
    def __init__(self,scenario, fully_obs=False,flat_actions=True,flat_obs=True):
        super().__init__()


        self.pengym_env = PenGymEnv(scenario, fully_obs, flat_actions, flat_obs)
        self.agent_ID= ["attacker","defender"]

        self.action_Spaces = {
            "attacker": self.pengym_env.action_space,
            "defender": Discrete(3)
        }

        self.observation_Spaces = {
            "attacker": self.pengym_env.observation_space,
            "defender": self.pengym_env.observation_space      #WATCH OUT check the actual implementation of the environment
        }
    @functools.lru_cache(None)
    def get_observation_space(self,agent):
        return self.observation_Spaces[agent]
    @functools.lru_cache(None)
    def get_action_space(self,agent):
        return self.action_Spaces[agent]
    
    def reset(self,seed=None,options=None):
        if seed is not None:
            self.np_random, self.np_random_seed = seeding.np_random(seed)
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agent_ID}
        self._cumulative_rewards = {agent: 0 for agent in self.agent_ID}
        self.terminations = {agent: False for agent in self.agent_ID}
        self.truncations = {agent: False for agent in self.agent_ID}
        obs,info= self.pengym_env.reset(seed=seed,options=options)
        self.state = obs
        self.infos["attacker"]=info
        self.agent_selectors = AgentSelector(self.agent_ID)
        self.selected_agent= self.agent_selectors.next()
        self.steps=0

    def step(self,action):

        if self.terminations[self.selected_agent] or self.truncations[self.selected_agent]:
            print("Epoch is over")
            self._was_dead_step(action=action)
            return
        
        else:
            if self.selected_agent=="attacker":
                obs,attacker_reward,terminated,truncated,attacker_info= self.pengym_env.step(action)
                self.state=obs
                self.rewards["attacker"]= attacker_reward
                self.rewards["defender"]= -attacker_reward
                self.terminations={a: terminated for a in self.agent_ID}
                self.truncations={a: truncated for a in self.agent_ID}
                self.infos["attacker"]=attacker_info

            elif self.selected_agent=="defender":
               print()
                

            self.selected_agent= self.agent_selectors.next()

    def config_PPO(self):
        ray.init(ignore_reinit_error=True)
        tune.register_env("PenGymMultiEnv-v0", lambda config: PenGymMultiEnv(config))

        config = (
            PPOConfig()
            .environment("PenGymMultiEnv-v0")
            .framework("torch") # Use PyTorch
            .multi_agent(
                policies={
                    # Format: (policy_class, obs_space, act_space, config_overrides)
                    # None defaults to the algorithm's standard policy (PPO in this case)
                    "attacker_policy": (None, self.observation_Spaces["attacker"], self.action_Spaces["attacker"], {}),
                    "defender_policy": (None, self.observation_Spaces["defender"], self.action_Spaces["defender"], {}),
                },
                # Map the agent string ID from the environment to the specific policy name
                policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: 
                    "attacker_policy" if agent_id == "attacker" else "defender_policy"
            )
    .training(
        train_batch_size=1000, 
        sgd_minibatch_size=128
    )
    .rollouts(num_rollout_workers=1) # distributed training
)
        algo=config.build()

    #def execute_PPO(self):


    def close(self):
        self.pengym_env.close()
        ray.shutdown


