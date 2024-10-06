import gym 

env = gym.make("Taxi-v3") # Taxi environment

observation = env.reset() # Reset the environment to the initial state

agent = load_agent() 

for step in range(100):
    action = agent(observation) # Choose an action based on the current observation
    observation, reward, done, info = env.step(action) # Perform the action and observe the next state, reward, and other info