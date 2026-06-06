import subprocess
import logging
from pengym.storyboard import Storyboard
import pengym.utilities as utils

class BlueActionExecutor:
    def __init__(self, config):
        self.config = config

    def execute_action(self, action_type, target_host_ip, kwargs={}):
        """Dispatcher for Blue Agent actions"""
        if action_type == Storyboard.CHECK_STATUS:
            return utils.check_status()
        elif action_type == Storyboard.BLOCK_CONNECTIONS:
            return utils.block_connections(target_host_ip)
        elif action_type == Storyboard.ISOLATE_HOST:
            return utils.do_isolate_host(target_host_ip,)


