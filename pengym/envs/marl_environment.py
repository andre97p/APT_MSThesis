import gymnasium as gym
from gymnasium.utils import seeding
from gymnasium.spaces import Discrete
from environment import PenGymEnv
import pengym.utilities as utils
import random
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict
from typing import Tuple

class PenGymMultiEnv(MultiAgentEnv):

    """
    A PettingZoo extension of Pengym environment paradigm
    """
    def __init__(self, environment):
        super().__init__()

        scenario = environment.get("scenario_name", "medium-multi-site")
        fully_obs = environment.get("fully_obs", False)
        flat_actions = environment.get("flat_actions", True)
        flat_obs = environment.get("flat_obs", True)
        self.pengym_env = PenGymEnv(scenario, fully_obs, flat_actions, flat_obs)
        self.agents = self.possible_agents = ["attacker", "defender"]

        self.action_spaces = {
            "attacker": self.pengym_env.action_space,
            "defender": Discrete(3)
        }

        self.observation_spaces = {
            "attacker": self.pengym_env.observation_space,
            "defender": self.pengym_env.observation_space      #WATCH OUT check the actual implementation of the environment
        }

    
    def reset(self,*,seed=None,options=None)-> Tuple[MultiAgentDict, MultiAgentDict]:
        obs,info = self.pengym_env.reset(seed=seed,options=options)

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.state = obs
        self.infos={
            "attacker":info,
            "defender":info
        }
        self.current_agent = random.choice(self.agents)
        self.steps=0
        self.state: MultiAgentDict = {self.current_agent: obs}
        self.infos: MultiAgentDict = {
            "attacker": info if self.current_agent == "attacker" else {},
            "defender": info if self.current_agent == "defender" else {}
        }
        return self.state,{self.current_agent:info}
    
    def step(self,action_dict)-> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        self.state: MultiAgentDict
        action=action_dict.get(self.current_agent)

        if self.terminations.get(self.current_agent) or self.truncations.get(self.current_agent):
            print("Epoch is over")
            pass
        
        else:
            if self.current_agent=="attacker":
                print("Red Agent Turn")
                obs, attacker_reward,terminated,truncated,attacker_info= self.pengym_env.step(action)
                self.state["attacker","defender"]=obs #to be confirmed
                self.rewards["attacker"]= attacker_reward
                self.rewards["defender"]= -attacker_reward
                self.terminations={a: terminated for a in self.agents}
                self.truncations={a: truncated for a in self.agents}
                self.infos["attacker"]=attacker_info
                self.current_agent="defender"

            elif self.current_agent=="defender":
               print("Blue Agent Turn")
               """
               """
               
               self.current_agent="attacker"

            self.steps+=1    
        return (self.state,self.rewards, self.terminations, self.truncations, self.infos)
    
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
                    #"attacker_policy": (None, self.observation_spaces["attacker"], self.action_spaces["attacker"], {}),
                    #"defender_policy": (None, self.observation_spaces["defender"], self.action_spaces["defender"], {}),
                },
                # Map the agent string ID from the environment to the specific policy name
                #policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: 
                 #   "attacker_policy" if agent_id == "attacker" else "defender_policy"
            )
    .training(
        train_batch_size=1000, 
        sgd_minibatch_size=128
    )
    .rollouts(num_rollout_workers=1) # distributed training
)
        

    #def execute_PPO(self):
        #algo = config.build()

    def close(self):
        self.pengym_env.close()
        ray.shutdown    


