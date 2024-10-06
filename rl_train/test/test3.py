import gym  
import time 

env = gym.make('CartPole-v0')  
env.reset()  
for _ in range(1000):
    env.render()  
    action = env.action_space.sample() 
    print(action)
    observation, reward, done, info = env.step(action)
    print(observation)
    time.sleep(0.1)
env.close()    
