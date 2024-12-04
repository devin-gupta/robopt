import gymnasium as gym
from gymnasium import spaces
import numpy as np
from envs.purchasing import PurchaseEnv

model = None

def load_model(logdir):
    global model
    if model is None:
        import os
        from stable_baselines3 import PPO

        model_path = os.path.join(logdir, "best_model.zip")
        assert os.path.exists(model_path), f"Model not found at {model_path}"
        model = PPO.load(model_path)

class MultiAgent_PurchaseEnv(gym.Env):

    class Task:
        def __init__(self):
            self.prize = np.random.randint(5, 10)

    class Smart_Bidder():
        def __init__(self, task_prize):
            load_model(logdir="./runs/baselines")
            self.distance = np.random.randint(1, 4)
            action, _ = model.predict({'prize': task_prize, 'distance': self.distance})
            self.bid = int(action)

            assert type(self.bid) == int, f"Expected type int, got {type(self.bid)} with action type {type(action)}"

    def __init__(self, num_bidders=3):
        super(MultiAgent_PurchaseEnv, self).__init__()
        self.task = self.Task()
        self.bidders = [self.Smart_Bidder(self.task.prize) for _ in range(num_bidders)]
        self.distance = np.random.randint(1, 4)
        self.action_space = spaces.Discrete(15)
        self.observation_space = spaces.Dict({
            'prize': spaces.Discrete(15),
            'distance': spaces.Discrete(4)
        })

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.task = self.Task()
        self.bidders = [self.Smart_Bidder(self.task.prize) for _ in range(len(self.bidders))]
        self.distance = np.random.randint(1, 4)
        initial_observation = {'prize': self.task.prize, 'distance': self.distance}
        return initial_observation, {}
    
    def get_reward(self, action):

        bids = [bidder.bid for bidder in self.bidders] + [action]

        # for bid in bids:
        #     assert type(bid) in (int, float), f"Expected type int or float, got type {type(bid)} for bid {bid}"

        relevant_bids = [
            bid for bid, bidder in zip(bids, self.bidders + [None])
            if bid <= (self.task.prize - (bidder.distance if bidder else self.distance))
        ]

        if relevant_bids:
            max_bid = max(relevant_bids)
            if action == max_bid:
                return self.task.prize - self.distance
            elif action > self.task.prize:
                return -1
            else:
                return 0
        else:
            return -1 if action > self.task.prize else 0

    def step(self, action):
        reward = 0
        terminated = True
        truncated = False
        info = {}
        reward = self.get_reward(action)
        return {'prize': self.task.prize, 'distance': self.distance}, reward, terminated, truncated, info

    def render(self, mode='human'):
        print(f"Task prize: {self.task.prize}")
        print(f"Bidders' bids: {[bidder.bid for bidder in self.bidders]}")
        print(f"Bidders' distances: {[bidder.distance for bidder in self.bidders]}")
        print(f"Distance: {self.distance}")