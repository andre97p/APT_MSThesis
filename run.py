
#############################################################################
# Run demo of PenGym functionality
#############################################################################

import time
import pengym
import numpy as np
import logging
import sys
import getopt
import pengym.utilities as utils
import pengym.training as training
from pengym.storyboard import Storyboard
import pengym.envs.training_marl as training_marl

storyboard = Storyboard()

#############################################################################
# Constants
#############################################################################

# Agent types
AGENT_TYPE_RANDOM = "random"
AGENT_TYPE_DETERMINISTIC = "deterministic"
DEFAULT_AGENT_TYPE = AGENT_TYPE_DETERMINISTIC

# Other constants
N_EPISODES=4000
MAX_STEPS = 150 # Max number of pentesting steps (sys.maxsize to disable)
RENDER_OBS_STATE = False

#############################################################################
# Functions
#############################################################################

# Select an action from the action space based on its name
# 'action_name' and its target 'action_target'
def select_action(action_space, action_name, action_target):
    for i in range(0, action_space.n):
        action = action_space.get_action(i)
        if action.name == action_name and action.target == action_target:
            return action

#############################################################################
# Create PenGym environment using scenario 'scenario_name'
def create_pengym_environment(scenario_name):
    env = pengym.create_environment(scenario_name)

    # Initialize seed for numpy (used to determine exploit success/failure) and
    # for the environment action space (used to determine order of random actions)
    seed = 1 # NORMAL: No e_ssh failure during pentesting path
    #seed = 300 # INCOMPLETE: Cause e_ssh failure during pentesting path
    np.random.seed(seed)
    env.action_space.seed(1)

    return env


# Create PenGym environment using custom scenario
def create_pengym_custom_environment(scenario_path):
    env = pengym.load(scenario_path)

    seed = 1
    np.random.seed(seed)
    env.action_space.seed(1)

    return env

# Print usage information
def usage():
    print("\nOVERVIEW: Run demo of the PenGym training framework for pentesting agents\n")
    print("USAGE: python3 run.py [options] <CONFIG_FILE> \n")
    print("OPTIONS:")
    print("-h, --help                     Display this help message and exit")
    print("-a, --agent_type <AGENT_TYPE>  Agent type (random/deterministic)")
    print("-d, --disable_pengym           Disable PenGym execution in cyber range")
    print("-n, --nasim_simulation         Enable NASim simulation execution")

#############################################################################
# Main program
#############################################################################
def main(args):

    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format='* %(levelname)s: %(filename)s: %(message)s')


    print("#########################################################################")
    print("PenGym: Pentesting Training Framework for Reinforcement Learning Agents")
    print("#########################################################################")

    # Default argument values
    agent_type = DEFAULT_AGENT_TYPE
    config_path = None

    # Parse command line arguments
    try:
        # Make sure to add ':' for short-form and '=' for long-form options that require an argument
        opts, trailing_args = getopt.getopt(args, "ha:dn",
                                            ["help", "agent_type=", "disable_pengym", "nasim_simulation"])
    except getopt.GetoptError as err:
        logging.error(f"Command-line argument error: {str(err)}")
        usage()
        sys.exit(1)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt in ("-a", "--agent"):
            agent_type = arg
        elif opt in ("-d", "--disable_pengym"):
            utils.ENABLE_PENGYM = False
        elif opt in ("-n", "--nasim_simulation"):
            utils.ENABLE_NASIM = True
        else:
            # Nothing to do, since unrecognized options are caught by
            # the getopt.GetoptError exception above
            pass

    # Get path of configuration file
    try:
        config_path = trailing_args[0]
    except Exception as e:
        logging.error(f"Configuration file is not specified")
        usage()
        sys.exit(2)

    # Print parameters
    print(f"* Execution parameters:")
    print(f"  - Agent type: {agent_type}")
    print(f"  - PenGym cyber range execution enabled: {utils.ENABLE_PENGYM}")
    print(f"  - NASim simulation execution enabled: {utils.ENABLE_NASIM}")

    # Check execution parameters
    if not (utils.ENABLE_PENGYM or utils.ENABLE_NASIM):
        logging.error("Either PenGym or NASim must be enabled")
        usage()
        sys.exit(2)

    print(f"* Read configuration from '{config_path}'...")
    utils.init_config_info(config_path)

    # Create an experiment environment using scenario path
    scenario_path = utils.replace_file_path(utils.config_info, storyboard.SCENARIO_FILE)
    print(f"* Create environment using custom scenario from '{scenario_path}'...")
    env = create_pengym_environment("tiny")
    """
    avg_reward = []
    for number in range(5):  #five agents is the number considered in the experiment
        if utils.ENABLE_PENGYM:
            print(f"* Read configuration from '{config_path}'...")
            utils.init_config_info(config_path)
            
            print("* Initialize MSF RPC client...")
            utils.init_msfrpc_client()
            
            print("* Initialize Nmap Scanner...")
            utils.init_nmap_scanner()
            
            # Create host map dictionary
            range_detail_file = utils.replace_file_path(database=utils.config_info,
                                                        file_name=storyboard.RANGE_DETAILS_FILE)

            utils.init_host_map(range_details_file=range_detail_file)

            # Initializer map of service ports
            utils.init_service_port_map()
        
            # Deactivate bridge that not connected to Internet
            utils.init_bridge_setup(range_details_file=range_detail_file)
            
        print(f"* Starting to train the agent {number+1} on the custom cyber range")
        agent = training.RedAgent(env,learning_rate=0.01, initial_epsilon=1, epsilon_decay=1 / (N_EPISODES / 2), final_epsilon=0.05, discount_factor=0.99)
        avg_reward.append(agent.training_agent_buffer(300,200,32))

        print("* Clean up MSF RPC client...")
        utils.cleanup_msfrpc_client()
        print("* Restore the to intial state of the firewalls for all hosts...")
        utils.save_restore_firewall_rules_all_hosts(flag=storyboard.RESTORE)
        avg_reward = np.mean(avg_reward,0)
        training.plot_rewards(avg_reward,1)
    """
    if utils.ENABLE_PENGYM:
            print(f"* Starting MARL training on the custom cyber range")

            
            # Execute the multi-agent training
            training_marl.execute_training(algo_type="mappo", training_iterations=300)
            
            print("* Clean up MSF RPC client...")
            utils.cleanup_msfrpc_client()
            print("* Restore the to intial state of the firewalls for all hosts...")
            utils.save_restore_firewall_rules_all_hosts(flag=storyboard.RESTORE)
            # ...


#############################################################################
# Run program
if __name__ == "__main__":
    start = time.time()
    main(sys.argv[1:])
    end = time.time()
    print(f"Execution Time: {end-start:1.6f}s")