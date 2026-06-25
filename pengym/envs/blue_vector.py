import subprocess
import logging
from pengym.storyboard import Storyboard
import pengym.utilities as utils
import math

class BlueActionExecutor:
    def __init__(self, config):
        self.config = config
        self.result: int
        self.action_reward= {
            "CHECK_STATUS":3,
            "BLOCK_CONNECTIONS":6,
            "ISOLATE_HOST":30,
            "DO_NOTHING":0


        }
        self.action_cost= {
            "CHECK_STATUS":1,
            "BLOCK_CONNECTIONS":3,
            "ISOLATE_HOST":20,
            "DO_NOTHING":1

            
        }

    def execute_action(self, action_type, target_host_ip, timestep, kwargs={}):
        """Dispatcher for Blue Agent actions"""
        result = 0
        if action_type == Storyboard.CHECK_STATUS:
            result = utils.check_status()
        elif action_type == Storyboard.BLOCK_CONNECTIONS:
            result = utils.block_connections(target_host_ip)
        elif action_type == Storyboard.ISOLATE_HOST:
            result = utils.do_isolate_host(target_host_ip)
        elif action_type == Storyboard.DO_NOTHING:
            result = utils.do_nothing()
        reward = self.compute_reward(action_type, result, timestep)
        return int(reward)

    def compute_reward(self, action_type, result, delta_t):
        """Compute reward using a time-penalised Bernoulli scheme.

        Check_status: R=3,C=1 | Block_connections: R=6,C=3
        Isolate_host: R=30,C=20 | Do_Nothing: R=0,C=1
        """
        beta = 0.01
        mu = min(100, math.exp(beta * delta_t))
        actual_reward=self.action_reward[action_type] if result==True else 0
        formula = (actual_reward - self.action_cost[action_type]) - mu
        
        return formula



