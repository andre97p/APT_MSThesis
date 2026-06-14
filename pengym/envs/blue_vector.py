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
        result=0
        """Dispatcher for Blue Agent actions"""
        if action_type == Storyboard.CHECK_STATUS:
            result= utils.check_status()
        elif action_type == Storyboard.BLOCK_CONNECTIONS:
            result= utils.block_connections(target_host_ip)
        elif action_type == Storyboard.ISOLATE_HOST:
            result= utils.do_isolate_host(target_host_ip,)
        elif action_type == Storyboard.DO_NOTHING:
            result= utils.do_isolate_host(target_host_ip,)
        reward= self.compute_reward(action_type,result,timestep)
        return int(reward)


    def compute_reward(self,action_type,result,delta_t):
        """
        Compute the rewards based on the Bernoulli distribution...
        The designed values for the specific actions are the following:
        Check_status: R=3,C=1
        Isolate_hosts: R=50,C=30
        Block_connections:R=6,C=3
        Do_Nothing:R=0,C:1
        """
        mu= min(100, math.exp(0.01*delta_t))
        actual_reward=self.action_reward[action_type] if result==True else 0
        formula = (actual_reward - self.action_cost[action_type]) - mu
        
        return formula



