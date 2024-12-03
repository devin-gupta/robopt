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
            robot_type = ['A-humanbot', 'B-navbot', 'C-embedbot']
            return f"Robot at ({self.x}, {self.y}) with type {robot_type[self.type]}"

    class Task:
        def __init__(self, grid_size):
            self.x = np.random.randint(0, grid_size[0])
            self.y = np.random.randint(0, grid_size[1])
            self.prize = np.random.randint(1, 100)
            self.type = np.random.choice([0, 1, 2]) # corresponding to ('manipulation', 'transport', 'specialty')

        def __str__(self):
            task_type = ['A-manipulation', 'B-transport', 'C-specialty']
            return f"Task at ({self.x}, {self.y}) with prize {self.prize} and type {task_type[self.type]}"


    def __init__(self, render_mode=None):
        super(BiddingEnv, self).__init__()

        self.grid_size = (10, 10)
        self.fig, self.ax = None, None
        self.robot_patches, self.task_patches = [], []

        self.n_robots = 9
        self.n_tasks = 6

        self.robots = [self.Robot(self.grid_size) for _ in range(self.n_robots)]
        self.tasks = [self.Task(self.grid_size) for _ in range(self.n_tasks)]

        self.bidding_matrix = np.zeros((self.n_robots, self.n_tasks), dtype=np.int32)

        self.current_step = 0
        self.max_step = 1

        self.final_multiplier = 100

        self.render_mode = render_mode

        # action space is discrete bidding matrix where bids are int [0, 10]
        self.action_space = spaces.Box(
            low=0,
            high=10,
            shape=(self.n_robots, self.n_tasks),
            dtype=np.int32
        )

        # observation space has robots and tasks
        self.observation_space = spaces.Dict({
            "robot_positions": spaces.Box(low=0, high=self.grid_size[0]-1, shape=(self.n_robots, 2), dtype=np.int32),
            "robot_types": spaces.Box(low=0, high=2, shape=(self.n_robots,), dtype=np.int32),
            "task_positions": spaces.Box(low=0, high=self.grid_size[0]-1, shape=(self.n_tasks, 2), dtype=np.int32),
            "task_prizes": spaces.Box(low=1, high=100, shape=(self.n_tasks,), dtype=np.int32),
            "task_types": spaces.Box(low=0, high=2, shape=(self.n_tasks,), dtype=np.int32)
        })
        
        '''
        OLD OBSERVATION SPACE
        observation_space_dict = {}

        for i in range(self.n_robots):
            observation_space_dict[f"robot_{i}_self_state"] = spaces.Box(
                low=np.array([0, 0, 0], dtype=np.float32),
                high=np.array([self.grid_size[0], self.grid_size[1], 2], dtype=np.float32),
                dtype=np.float32
            )
            observation_space_dict[f"robot_{i}_bidding_matrix"] = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.n_robots, self.n_tasks),
                dtype=np.float32
            )

        # Assign the observation space
        self.observation_space = spaces.Dict(observation_space_dict)
        '''

    def get_cost(self, robot, task):
        distance = np.linalg.norm(np.array([robot.x, robot.y]) - np.array([task.x, task.y]))
        type_match = 0 if round(robot.type) == round(task.type) else 1
        return distance * 0.5 * (1 + type_match)
        # distance (max of 17.1, min of 0) * 0.5 * 33% * (1) + 66% (2)
        # 14 * 0.5 * (1.6) = 9.5
    
    def observe(self):
        robot_positions = np.array([[robot.x, robot.y] for robot in self.robots], dtype=np.int32)
        robot_types = np.array([robot.type for robot in self.robots], dtype=np.int32)
        task_positions = np.array([[task.x, task.y] for task in self.tasks], dtype=np.int32)
        task_prizes = np.array([task.prize for task in self.tasks], dtype=np.int32)
        task_types = np.array([task.type for task in self.tasks], dtype=np.int32)

        return {
            "robot_positions": robot_positions,
            "robot_types": robot_types,
            "task_positions": task_positions,
            "task_prizes": task_prizes,
            "task_types": task_types
        }


    def step(self, action):

        assert action.shape == (self.n_robots, self.n_tasks), f"Invalid action shape: {action.shape}"
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        print('bidding matrix shape', self.bidding_matrix.shape, ' and action shape ', action.shape)

        self.bidding_matrix = action

        assert self.bidding_matrix.shape == (self.n_robots, self.n_tasks), f"Invalid action shape: {self.bidding_matrix.shape}"

        # Increment step count
        self.current_step += 1
        done = self.current_step >= self.max_step

        # Calculate reward
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
                # reward += E[0, 100] - 9.5

        # Scale reward based on progress
        total_possible_reward = len(self.tasks) # reward is per task

        reward = reward / (len(self.tasks) * self.optimal_reward())
        
        '''
        if not done:
            reward /= total_possible_reward
        else:
            reward = self.final_multiplier * reward / total_possible_reward
        '''

        observation = self.observe()

        # Additional info dictionary (can be expanded as needed)
        info = {}

        # Truncated is False since termination is based only on `done`
        truncated = False

        return observation, reward, done, truncated, info


    def reset(self, seed=1, options=None):
        # Seed the environment for reproducibility
        if seed is not None:
            super().reset(seed=seed)  # Ensures compatibility with Gym's seeding
            np.random.seed(seed)

        # Randomly initialize robots and tasks
        self.robots = [self.Robot(self.grid_size) for _ in range(self.n_robots)]
        self.tasks = [self.Task(self.grid_size) for _ in range(self.n_tasks)]

        # Reset the bidding matrix
        self.bidding_matrix = np.zeros((self.n_robots, self.n_tasks), dtype=np.int32)
        assert type(self.bidding_matrix[0, 0]) == np.int32, f"Invalid bidding matrix type: {type(self.bidding_matrix[0, 0])}"

        # Reset step counter
        self.current_step = 0

        # Get initial observation
        observation = self.observe()

        # Return observation and empty info dictionary
        return observation, {}

    def render(self, mode=None):
        """
        Render the environment state.
        
        Supported render modes:
        - None: No rendering
        - 'human': Text-based verbose output
        - 'rgb_array': Matplotlib plot as an RGB array
        - 'ansi': Tabular representation of bids
        """
        if mode is None:
            return

        if mode == 'human':
            print(
                tabulate(
                    self.bidding_matrix, 
                    headers=[f"Task {i+1}" for i in range(self.n_tasks)], 
                    tablefmt='fancy_grid'
                )
            )
            print("\n")
            return None
        
        elif mode == 'human_verbose':
            # Verbose text output
            print(f"Step: {self.current_step}")
            print("\nRobots:")
            for i, robot in enumerate(self.robots):
                print(f"  Robot {i + 1}: {robot}")
            print("\nTasks:")
            for i, task in enumerate(self.tasks):
                print(f"  Task {i + 1}: {task}")
            print("\nBidding Matrix:")
            print(
                tabulate(
                    self.bidding_matrix, 
                    headers=[f"Task {i+1}" for i in range(self.n_tasks)], 
                    tablefmt='fancy_grid'
                )
            )
            print("\n")
            return None

        elif mode == 'rgb_array':
            # Create a plot and return it as an RGB array
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

            # Convert plot to RGB array
            self.fig.canvas.draw()
            image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype='uint32')
            image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            
            plt.close(self.fig)  # Close the figure to free up memory
            self.fig = None
            self.ax = None
            
            return image

        elif mode == 'ansi':
            # Tabular representation of bids
            return tabulate(
                self.bidding_matrix, 
                headers=[f"Task {i+1}" for i in range(self.n_tasks)], 
                tablefmt='fancy_grid'
            )

        else:
            raise ValueError(f"Unsupported render mode: {mode}. Choose from None, 'human', 'rgb_array', or 'ansi'.")
        
    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

    def optimal_reward(self):

        optimal_reward = 0

        for task in self.tasks:

            min_cost = float('inf')

            for robot in self.robots:
            
                cost = self.get_cost(robot, task)

                if cost < min_cost:
                    min_cost = cost

            optimal_reward += task.prize - min_cost

        optimal_reward = optimal_reward / (len(self.tasks))

        return optimal_reward