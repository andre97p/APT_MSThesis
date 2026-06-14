import gymnasium as gym
from gymnasium.utils import seeding
from gymnasium.spaces import Discrete,Space
from environment import PenGymEnv
import random
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict
from ray.rllib.policy.policy import PolicySpec
from typing import Tuple,Dict
import utilities as utils
from storyboard import Storyboard
from blue_vector import BlueActionExecutor
from typing import cast

class PenGymMultiEnv(MultiAgentEnv):

    """
    A Multi-Agent extension of Pengym environment
    """
    action_Space: Dict[str, Space]
    observation_Space: Dict[str, Space]
    def __init__(self, environment):
        super().__init__()

        scenario = environment.get("scenario_name", "medium-multi-site")
        fully_obs = environment.get("fully_obs", False)
        flat_actions = environment.get("flat_actions", True)
        flat_obs = environment.get("flat_obs", True)
        self.pengym_env = PenGymEnv(scenario, fully_obs, flat_actions, flat_obs)
        self.agents = self.possible_agents = ["attacker", "defender"]
        self.blue_executor= BlueActionExecutor(config=environment)
        self.action_Space = {
            "attacker": self.pengym_env.action_space,
            "defender": Discrete(4)
        }

        self.observation_Space = {
            "attacker": self.pengym_env.observation_space,
            "defender": self.pengym_env.observation_space      #WATCH OUT check the actual implementation of the environment
        }

    
    def reset(self,*,seed=None,options=None)-> Tuple[MultiAgentDict, MultiAgentDict]:
        obs,info = self.pengym_env.reset(seed=seed,options=options)

        #Restoring the initial state of the environment
        restore_alert=utils.clean_alertList()
        restore_isolated= utils.restore_isolated()
        restore_connections= utils.unlock_connections()
        print(f"Cleaning alerts list gives {restore_alert} result")
        print(f"Restoring shutted down connections gives {restore_isolated} result")
        print(f"Restoring connection filters gives {restore_connections} result")

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
    

    def step(self, action_dict) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        opponent = "attacker" if self.current_agent == "defender" else "defender"
        action = action_dict.get(self.current_agent)
        rewards = {a: 0 for a in self.agents}
        terminations = {"__all__": False}
        truncations = {"__all__": False}
        infos = {}

        # Handle episode boundaries
        if self.terminations.get(self.current_agent) or self.truncations.get(self.current_agent):
            print("Epoch is over")
            terminations["__all__"] = True
            return (cast(MultiAgentDict, {}),
            cast(MultiAgentDict, rewards),
            cast(MultiAgentDict, terminations),
            cast(MultiAgentDict, truncations),
            cast(MultiAgentDict, infos))
        
        if self.current_agent == "attacker":
            print("Red Agent Turn")
            obs, attacker_reward, terminated, truncated, attacker_info = self.pengym_env.step(action)
            
            self.last_obs = obs 
            
            rewards["attacker"] = attacker_reward
            rewards["defender"] = -attacker_reward
            
            self.terminations = {a: terminated for a in self.agents}
            self.truncations = {a: truncated for a in self.agents}
            infos["attacker"] = attacker_info
            
            self.observation_Space["defender"]= self.last_obs

        elif self.current_agent == "defender":
            print("Blue Agent Turn")
            
            # Decode Action Space

            action_mapping = {
                0: getattr(Storyboard, "CHECK_STATUS"),
                1: getattr(Storyboard, "BLOCK_CONNECTIONS"),
                2: getattr(Storyboard, "ISOLATE_HOST"),
                3: getattr(Storyboard, "DO_NOTHING")
            }
    
            parsed_action = int(action) if action is not None else 3
            action_type = action_mapping.get(parsed_action, getattr(Storyboard, "DO_NOTHING"))
            
            target_ip = None
            if utils.alerts and len(utils.alerts) > 0:
                # Target the source IP of the latest anomalous flow for mitigation/isolation
                target_ip = utils.alerts[-1].get("src_ip")
 
            actual_reward = self.blue_executor.execute_action(action_type = action_type, target_host_ip=target_ip,timestep=self.steps)
            rewards["defender"]= actual_reward
            rewards["attacker"]= -actual_reward
            
            infos["defender"] = {
                "executed_action": action_type,
                "target_ip": target_ip,
                "success": rewards
            }
              
            self.observation_Space["attacker"]= self.last_obs

        # Sync dictionary structures to RLlib MultiAgent specs

        terminations["__all__"] = all(self.terminations.values())
        truncations["__all__"] = all(self.truncations.values())
        self.current_agent=opponent
        self.steps += 1

        return (cast(MultiAgentDict,self.last_obs),
            cast(MultiAgentDict, rewards),
            cast(MultiAgentDict, terminations),
            cast(MultiAgentDict, truncations),
            cast(MultiAgentDict, infos))
    

def close(self):
    self.pengym_env.close()
    ray.shutdown()


############################# TRAINING PROCEDURES ##################################  

temp_env= PenGymMultiEnv({})
obs_space_attacker = temp_env.observation_Space["attacker"]
act_space_attacker = temp_env.action_Space["attacker"]
obs_space_defender = temp_env.observation_Space["defender"]
act_space_defender = temp_env.action_Space["defender"]

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
    

def execute_training(self):
    algo = self.config_IPPO().build()
    while(True):
        pass