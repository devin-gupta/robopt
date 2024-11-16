# basic imports
import numpy as np
from collections import OrderedDict
from tabulate import tabulate

# gym imports
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv

# visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, Circle
from IPython.display import clear_output


class BiddingEnv(gym.Env):

    class Robot: 
        def __init__(self, grid_size):
            # Random x, y position within grid boundaries
            self.x = np.random.randint(0, grid_size[0])
            self.y = np.random.randint(0, grid_size[1])
            self.type = np.random.choice([0, 1, 2]) # corresponding to ('humanbot', 'navbot', 'embedbot')

        def __str__(self):
            robot_type = ['humanbot', 'navbot', 'embedbot']
            return f"Robot at ({self.x}, {self.y}) with type {robot_type[self.type]}"

    class Task:
        def __init__(self, grid_size):
            self.x = np.random.randint(0, grid_size[0])
            self.y = np.random.randint(0, grid_size[1])
            self.prize = np.random.randint(1, 4)
            self.type = np.random.choice([0, 1, 2]) # corresponding to ('manipulation', 'transport', 'specialty')

        def __str__(self):
            task_type = ['manipulation', 'transport', 'specialty']
            return f"Task at ({self.x}, {self.y}) with prize {self.prize} and type {task_type[self.type]}"


    def __init__(self):
        super(BiddingEnv, self).__init__()

        self.grid_size = (10, 10)
        self.fig, self.ax = None, None
        self.robot_patches, self.task_patches = [], []

        self.n_robots = 9
        self.n_tasks = 6

        self.robots = [self.Robot(self.grid_size) for _ in range(self.n_robots)]
        self.tasks = [self.Task(self.grid_size) for _ in range(self.n_tasks)]

        self.bidding_matrix = np.zeros((self.n_robots, self.n_tasks), dtype=np.int8)

        self.current_step = 0
        self.max_step = 10

        self.final_multiplier = 100

        # action space is discrete between 0 and 10 for each robot
        self.action_space = spaces.Box(
            low=0,
            high=10,
            shape=(self.n_robots, self.n_tasks),
            dtype=np.int8
        )

        # observation space is a dictionary of dictionaries, where each dictionary has two keys: self_state and bidding_matrix
        self.observation_space = spaces.Dict({
            f"robot_{i}": spaces.Dict({
                'self_state': spaces.Box(
                    # x, y, type
                    low=np.array([0, 0, 0], dtype=np.int8),
                    high=np.array([self.grid_size[0], self.grid_size[1], 2], dtype=np.int8),
                    dtype=np.int8
                ),
                'bidding_matrix': spaces.Box(
                    low=0,
                    high=10,
                    shape=(self.n_robots, self.n_tasks),
                    dtype=np.int8
                )
            }) for i in range(self.n_robots)
        })

    def get_cost(self, robot, task):
        distance = np.linalg.norm(np.array([robot.x, robot.y]) - np.array([task.x, task.y]))
        type_match = 1 if robot.type == task.type else 0
        return distance * 0.5 * (1 + type_match)
    
    def observe(self):

        observation = {
            f"robot_{i}": {
                'self_state': np.array([
                    robot.x,   # Existing x-coordinate
                    robot.y,   # Existing y-coordinate
                    robot.type # Existing type
                ], dtype=np.int8),
                'bidding_matrix': self.bidding_matrix  # Use the existing bidding matrix
            }
            for i, robot in enumerate(self.robots)
        }

        return observation

    def step(self, action):
        # action is the bidding matrix
        self.bidding_matrix = action 

        observation = self.observe()

        self.current_step += 1
        done = self.current_step >= self.max_step 

        # define reward function. 
        # for each task, if it is completed, add the prize to the reward
        reward = 0
        for task_index, task in enumerate(self.tasks):
            max_bid = 0
            max_bid_robot_index = None
            for robot_index in range(self.n_robots):
                if self.bidding_matrix[robot_index, task_index] > max_bid:
                    max_bid = self.bidding_matrix[robot_index, task_index]
                    max_bid_robot_index = robot_index

            if max_bid_robot_index is not None:
                reward += task.prize - self.get_cost(self.robots[max_bid_robot_index], task)

        # Scale reward
        if not done:
            reward = reward / (len(self.tasks) * self.max_step)
        else:
            reward = self.final_multiplier * reward / (len(self.tasks) * self.max_step)

        info = {}

        truncated = False

        return observation, reward, done, truncated, info

    def reset(self, seed=None, options=None):

        if seed is not None:
            np.random.seed(seed)

        # Randomly initialize robots and tasks
        self.robots = [self.Robot(self.grid_size) for _ in range(self.n_robots)]
        self.tasks = [self.Task(self.grid_size) for _ in range(self.n_tasks)]

        # Reset bidding matrix
        self.bidding_matrix = np.zeros((self.n_robots, self.n_tasks), dtype=np.int8)

        self.current_step = 0
        
        observation = self.observe()
        return observation, {}

    def render(self, mode='verbose'):
        if mode == 'verbose':
            # clear_output(wait=True)
            print(f"Step: {self.current_step}")
            print(*[f"{robot}\n" for robot in self.robots], sep='')
            print(*[f"{task}\n" for task in self.tasks], sep='')
            print(
                f"Bidding Matrix: \n"
                f"{tabulate(self.bidding_matrix, headers=[f'task {i+1}' for i in range(self.n_tasks)], tablefmt='fancy_grid')}"
            )

        if mode == 'bids':
            print(
                f"Bidding Matrix: \n"
                f"{tabulate(self.bidding_matrix, headers=[f'task {i+1}' for i in range(self.n_tasks)], tablefmt='fancy_grid')}"
            )
        
        elif mode == 'plot':
            if self.fig is None or self.ax is None:
                self.fig, self.ax = plt.subplots(figsize=(8, 8))
                self.ax.set_xlim(0, self.grid_size[0])
                self.ax.set_ylim(0, self.grid_size[1])
                self.ax.set_aspect('equal')
                self.ax.grid(True)

            # Clear previous patches
            for patch in self.robot_patches + self.task_patches:
                patch.remove()
            self.robot_patches.clear()
            self.task_patches.clear()

            # Draw robots
            for robot in self.robots:
                color = plt.cm.Set1(robot.type / 3)
                square = Rectangle((robot.x - 0.4, robot.y - 0.4), 0.8, 0.8,
                                fill=True, facecolor=color, edgecolor='black')
                self.ax.add_patch(square)
                self.robot_patches.append(square)

            # Draw tasks
            for task in self.tasks:
                circle = Circle((task.x, task.y), 0.5,
                              facecolor='blue', edgecolor='blue')
                self.ax.add_patch(circle)
                self.task_patches.append(circle)

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.1)

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

