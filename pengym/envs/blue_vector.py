import subprocess
import logging
from pengym.storyboard import Storyboard
import pengym.utilities as utils

class BlueActionExecutor:
    def __init__(self, config):
        self.config = config
        self.result: int
    

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
        return reward


    def compute_reward(action_type,result,timestep):
        """
        Compute the rewards based on the Bernoulli distribution...
        The designed values for the specific actions are the following:
        Check_status: R=3,C=1
        Isolate_hosts: R=50,C=30
        Block_connections:R=6,C=3
        Do_Nothing:R=0,C:1
        """
        return 0



