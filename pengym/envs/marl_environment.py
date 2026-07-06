import gymnasium as gym
from gymnasium.utils import seeding
from gymnasium.spaces import Discrete, Space, MultiBinary
from pengym.envs.environment import PenGymEnv
import random
import numpy as np
import ray
import os
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict
from ray.rllib.policy.policy import PolicySpec
from typing import Tuple
import pengym.utilities as utils
from pengym.storyboard import Storyboard
from pengym.envs.blue_vector import BlueActionExecutor
from typing import cast
from nasim.scenarios import make_benchmark_scenario

_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "CONFIG.yml")

class PenGymMultiEnv(MultiAgentEnv):

    """
    A Multi-Agent extension of Pengym environment
    """

    def __init__(self, scenario_name="tiny", env_config=None):
        super().__init__()

        env_config = env_config or {}
        scenario_name = env_config.get("scenario_name", scenario_name)

        config_path = os.path.abspath(_DEFAULT_CONFIG_PATH)

        utils.init_config_info(config_path)
        utils.init_service_port_map()

        # Execution backend: default to the emulated PenGym cyber range
        # (ENABLE_NASIM=False). Overridable through the RLlib env_config.
        utils.ENABLE_PENGYM = bool(env_config.get("enable_pengym", True))
        utils.ENABLE_NASIM = bool(env_config.get("enable_nasim", False))

        # The live cyber-range backend needs Metasploit, Nmap and the CyRIS
        # host map built from the range-details file (NOT CONFIG.yml).
        range_detail_file = None
        if utils.ENABLE_PENGYM:
            utils.init_msfrpc_client()
            utils.init_nmap_scanner()
            range_detail_file = utils.replace_file_path(
                database=utils.config_info,
                file_name=Storyboard.RANGE_DETAILS_FILE)
            utils.init_host_map(range_details_file=range_detail_file)

        scenario_obj = make_benchmark_scenario(scenario_name)

        self.pengym_env = PenGymEnv(scenario_obj, False, True, True)

        # Bridge setup relies on utils.scenario, which PenGymEnv sets above,
        # so it must run after the environment has been constructed.
        if utils.ENABLE_PENGYM:
            utils.init_bridge_setup(range_details_file=range_detail_file)

        self.agents = self.possible_agents = ["attacker", "defender"]
        self.blue_executor= BlueActionExecutor(self.pengym_env)
        self.action_Space = {
            "attacker": self.pengym_env.action_space,
            "defender": Discrete(4)
        }
        num_nodes = len(self.pengym_env.network.hosts)
        num_services = self.pengym_env.scenario.num_services
        num_os = self.pengym_env.scenario.num_os

        self.observation_Space = {
            "attacker": self.pengym_env.observation_space,
            "defender": gym.spaces.Dict({
                "topology": MultiBinary(num_nodes),
                "system_configs_services": MultiBinary(num_nodes * num_services),
                "system_configs_os": MultiBinary(num_nodes * num_os),
                "active_alerts": MultiBinary(num_nodes*10),
                "firewall_status": MultiBinary(num_nodes * num_services)
        })

        }
        self._attacker_obs = None
        self.steps = 0

        self.max_episode_steps = int(env_config.get("max_episode_steps", 100)) #100 is specific value for the tiny scenario

    def _get_defender_obs(self) -> dict:
        """
        Extracts and returns the defender observation as binary numpy arrays
        matching the defined observation space keys and shapes.
        """
        num_nodes = len(self.pengym_env.network.hosts)
        num_services = self.pengym_env.scenario.num_services
        num_os = self.pengym_env.scenario.num_os

        # Defender sees all hosts in the network
        topology = np.ones(num_nodes, dtype=np.int8)

        system_configs_services = np.zeros(num_nodes * num_services, dtype=np.int8)
        system_configs_os = np.zeros(num_nodes * num_os, dtype=np.int8)

        if hasattr(self.pengym_env, 'network') and hasattr(self.pengym_env, 'current_state'):
            for i, host_addr in enumerate(self.pengym_env.network.address_space):
                host_vector = self.pengym_env.current_state.get_host(host_addr)
                if host_vector.services:
                    for j, active in enumerate(host_vector.services.values()):
                        if j < num_services:
                            system_configs_services[i * num_services + j] = int(bool(active))
                if host_vector.os is not None:
                    os_active = (any(v for v in host_vector.os.values())
                                 if isinstance(host_vector.os, dict) else bool(host_vector.os))
                    system_configs_os[i * num_os] = int(os_active)

        # One bit per alert slot (up to 10 per node)
        active_alerts = np.zeros(num_nodes * 10, dtype=np.int8)
        alert_list = utils.alerts if hasattr(utils, 'alerts') else []
        for j in range(min(len(alert_list), num_nodes * 10)):
            active_alerts[j] = 1


        firewall_status = np.zeros(num_nodes * num_services, dtype=np.int8)
        isolated = utils.isolated_hosts if hasattr(utils, 'isolated_hosts') else {}
        for i, host_addr in enumerate(self.pengym_env.network.address_space):
            if str(host_addr) in isolated or host_addr in isolated:
                for j in range(num_services):
                    firewall_status[i * num_services + j] = 1

        return {
            "topology": topology,
            "system_configs_services": system_configs_services,
            "system_configs_os": system_configs_os,
            "active_alerts": active_alerts,
            "firewall_status": firewall_status,
        }
    def reset(self, *, seed=None, options=None) -> Tuple[MultiAgentDict, MultiAgentDict]:
        obs, info = self.pengym_env.reset(seed=seed, options=options)

        utils.clean_alertList()

        restore_isolated = utils.restore_isolated()
        restore_connections = utils.unlock_connections()
        print(f"Restoring shutted down connections gives {restore_isolated} result")
        print(f"Restoring connection filters gives {restore_connections} result")

        self.agents = self.possible_agents[:]
        self._attacker_obs = obs  # persist so defender-turn step can return it
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {"attacker": info, "defender": info}
        self.current_agent = random.choice(self.agents)
        self.steps = 0

        if self.current_agent == "attacker":
            agent_obs = {self.current_agent: obs}
        else:
            agent_obs = {self.current_agent: self._get_defender_obs()}

        return agent_obs, {self.current_agent: info}
    

    def step(self, action_dict) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        opponent = "attacker" if self.current_agent == "defender" else "defender"
        action = action_dict.get(self.current_agent)
        rewards = {a: 0 for a in self.agents}
        infos = {}


        if self.terminations.get(self.current_agent) or self.truncations.get(self.current_agent):
            print("Epoch is over")
            return self._episode_end_return(rewards, infos)

        if self.current_agent == "attacker":
            print("Red Agent Turn")
            obs, attacker_reward, terminated, truncated, attacker_info = self.pengym_env.step(int(action))
            #w=utils.regularize(0.01,self.steps,50)
            #attacker_reward=attacker_reward - w

            self._attacker_obs = obs  # persist so defender-turn step can return it

            print(f"Attacker Reward: {int(attacker_reward)}")
            rewards["attacker"] = (attacker_reward)
            rewards["defender"] = -(attacker_reward)

            # Rely on NASim's native goal condition for termination. 
            if terminated:
                print("Episode terminated: all sensitive hosts compromised "
                      f"{self.pengym_env.network.sensitive_addresses}")
            self.terminations = {a: terminated for a in self.agents}
            self.truncations = {a: truncated for a in self.agents}
            infos[opponent] = attacker_info

        elif self.current_agent == "defender":
            print("Blue Agent Turn")

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
                target_ip = utils.alerts[-1].get("src_ip")

            actual_reward = self.blue_executor.execute_action(
                action_type=action_type, target_host_ip=target_ip, timestep=self.steps)
            print(f"Defender Reward: {actual_reward}")
            rewards["defender"] = actual_reward
            rewards["attacker"] = -actual_reward

            infos[opponent] = {
                "executed_action": action_type,
                "target_ip": target_ip,
                "success": rewards
            }

        self.steps += 1


        if self.steps >= self.max_episode_steps and not all(self.terminations.values()):
            self.truncations = {a: True for a in self.agents}
            print(f"Episode truncated: reached max_episode_steps={self.max_episode_steps}")

        episode_over = all(self.terminations.values()) or all(self.truncations.values())

        if episode_over:

            return self._episode_end_return(rewards, infos)

        # Mid-episode: pass the turn to the opponent only.
        if opponent == "defender":
            next_obs: MultiAgentDict = {opponent: self._get_defender_obs()}
        else:
            next_obs = {opponent: self._attacker_obs}

        terminations = {a: self.terminations[a] for a in self.agents}
        terminations["__all__"] = False
        truncations = {a: self.truncations[a] for a in self.agents}
        truncations["__all__"] = False
        self.current_agent = opponent

        return (cast(MultiAgentDict, next_obs),
                cast(MultiAgentDict, rewards),
                cast(MultiAgentDict, terminations),
                cast(MultiAgentDict, truncations),
                cast(MultiAgentDict, infos))

    def _episode_end_return(self, rewards, infos):
        """
        Build the terminal step return for a finished episode.

        Both agents receive a final observation, their per-agent done flag and the
        `__all__` marker so RLlib can close out *both* trajectories in a single step,
        regardless of whose turn ended the episode.
        """
        next_obs: MultiAgentDict = {
            "attacker": self._attacker_obs,
            "defender": self._get_defender_obs(),
        }
        terminations = {a: bool(self.terminations.get(a, False)) for a in self.agents}
        truncations = {a: bool(self.truncations.get(a, False)) for a in self.agents}
        terminations["__all__"] = all(terminations[a] for a in self.agents)
        truncations["__all__"] = all(truncations[a] for a in self.agents)

        return (cast(MultiAgentDict, next_obs),
                cast(MultiAgentDict, rewards),
                cast(MultiAgentDict, terminations),
                cast(MultiAgentDict, truncations),
                cast(MultiAgentDict, infos))
    

    def close(self):
        self.pengym_env.close()
        ray.shutdown()
