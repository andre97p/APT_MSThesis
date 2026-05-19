import gymnasium as gym
from gymnasium.utils import seeding
from gymnasium.spaces import Discrete
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers, AgentSelector
from environment import PenGymEnv
import pengym.utilities as utils
import functools

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
            "defender": self.pengym_env.observation_space      #check the actual visibility
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

    def close(self):
        self.pengym_env.close()


