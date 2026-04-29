import numpy as np
import random
from collections import deque,defaultdict
import matplotlib.pyplot as plt
import gymnasium as gym


class RedAgent:

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):

        self.env= env
        self.q_table= {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

    def get_state_key(self,state):
        return str(state)

    def get_q_values(self,state_key, action_space_size):
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(action_space_size)
        return self.q_table[state_key]

    def get_action(self, obs) -> int:
        # With probability epsilon: explore (random action)
        if np.random.random() < self.epsilon:
            return int(self.env.action_space.sample())
        # With probability (1-epsilon): exploit (best known action)
        else:
            return int(np.argmax(self.q_table[obs]))
    
    def update(
            self,
            state,
            action: int,
            reward: float,
            terminated: bool,
            next_state
    ):
        current_q_values = self.get_q_values(state, 18)
        next_q_values_array = self.get_q_values(next_state, 18)
        old_q_value = current_q_values[action]
        next_q_values = (not terminated) * np.max(next_q_values_array)
                
#The Bellman Equation (Q-Learning Update Rule)
        new_q_value = old_q_value + self.learning_rate * (reward + self.discount_factor * next_q_values - old_q_value)
        self.q_table[state][action] = new_q_value

    def training_agent(self,episodes):
            rewards=[]
            print("Initiating Q-Learning Agent Training...")
            for episode in range(episodes):
                state, info = self.env.reset()
                state= self.get_state_key(state)
                total_reward = 0
                done=False
                step=0
                while not done or step==1000:
                    action= self.get_action(state)
                    #Execute the action in the environment
                    next_state, reward, terminated, truncated, info = self.env.step(action)
                    next_state = self.get_state_key(next_state)
                    reward= float(reward)
                    # Check if the episode is over (either agent won/failed, or hit time limit)
                    done = terminated or truncated
                    self.update(state,action,reward,terminated,next_state)             
                    
                    state = next_state
                    total_reward += reward
                    step+=1
                    
                    if done: 
                        rewards.append(total_reward) 
                        break
                            
                self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)
                
                # Print progress every 100 episodes
                if (episode + 1) % 100 == 0:
                    print(f"Episode {episode + 1} | Reward: {total_reward} | Epsilon: {self.epsilon:.2f}")
            print("Training Complete. Agent is ready for deployment.")
            return rewards

    def training_agent_buffer(self, episodes,capacity,batch_size):
        rewards=[]
        replay_buffer=deque(maxlen=capacity)

        def store_experience(curr_s,action,reward,next_s,done):
            new_el=(curr_s,action,reward,next_s,done)
            replay_buffer.append(new_el)

        def sample_experience(quantity):
            return random.sample(replay_buffer,quantity)
        
        print("Initiating Q-Learning Agent with Replay buffer Training...")
        for episode in range(episodes):
            state, info = self.env.reset()
            state= self.get_state_key(state)
            total_reward = 0
            done=False
            while not done:

                action=self.get_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                reward= float(reward)
                done = terminated or truncated
                store_experience(state, action, reward, next_state, done)

                if len(replay_buffer) >= batch_size:
                    mini_batch = sample_experience(batch_size)
                    for b_state, b_action, b_reward, b_next_state, b_done in mini_batch:
                        self.update(b_state,b_action,b_reward,b_done,b_next_state)
                
                state = next_state
                total_reward += reward
            
                if done: 
                    rewards.append(total_reward) 
                    break
                        
            self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)
            
            # Print progress every 100 episodes
            if (episode + 1) % episodes == 0:
                print(f"Episode {episode + 1} | Rewards: {total_reward} | Epsilon: {self.epsilon:.2f}")
        print("Training Complete. Agent is ready for deployment.")
        return rewards


def get_moving_avgs(arr, window, convolution_mode):
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    )/window

def plot_rewards(rewards,rolling_lenght):
    plt.figure(figsize=(100, 50))
    '''clean_rewards= get_moving_avgs(rewards,
                                   rolling_lenght,
                                   "valid"
    )'''
    plt.title('PenGym Agent Training Progress (Q-Learning)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.plot(range(len(rewards)),rewards, alpha=0.8, color='blue', label='Pengym Rewards')
    plt.show()