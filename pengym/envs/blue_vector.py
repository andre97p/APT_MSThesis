import subprocess
import logging
from pengym.storyboard import Storyboard

class BlueActionExecutor:
    def __init__(self, config):
        self.config = config

    def execute_action(self, action_type, target_host_ip, kwargs={}):
        """Dispatcher for Blue Agent actions"""
        if action_type == Storyboard.CHECK_STATUS:
            return self.do_check_status(target_host_ip)
        elif action_type == Storyboard.BLOCK_CONNECTIONS:
            return self.do_block_connection(target_host_ip, kwargs.get('attacker_ip'))
        elif action_type == Storyboard.ISOLATE_HOST:
            return self.do_isolate_host(target_host_ip)

    def do_check_status(self, host_ip):
        """Checks for suspicious processes (e.g., reverse shells, unauthorized cron jobs)."""
        logging.info(f"[Blue] Checking status on {host_ip}")
        # Command checks for common reverse shell processes or high CPU spikes
        cmd = f"ssh vagrant@{host_ip} 'ps -ef | grep -E \"nc -e|bash -i|meterpreter\" | grep -v grep'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        is_compromised = len(result.stdout.strip()) > 0
        return is_compromised

    def do_block_connection(self, host_ip, attacker_ip):
        """Blocks connections from a specific IP using iptables."""
        logging.info(f"[Blue] Blocking IP {attacker_ip} on host {host_ip}")
        # Note: In a real PenGym environment, this might call a modified version of add_firewall_rule.exp
        cmd = f"ssh vagrant@{host_ip} 'sudo iptables -A INPUT -s {attacker_ip} -j DROP'"
        result = subprocess.run(cmd, shell=True)
        return result.returncode == 0

    def do_isolate_host(self, host_ip):
        """Isolates the host by routing all traffic to a null interface or dropping the gateway."""
        logging.info(f"[Blue] Isolating host {host_ip}")
        cmd = f"ssh vagrant@{host_ip} 'sudo ip link set eth1 down'" #TO BE CHANGED WITH THE RIGHT INTERFACE
        result = subprocess.run(cmd, shell=True)
        return result.returncode == 0