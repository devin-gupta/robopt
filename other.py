import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers

np.random.seed(1024)

class Task:
    def __init__(self, x, y):
        self.name = "Task"
        self.x = x
        self.y = y
        self.reward = np.random.randint(0, 100)

    def __str__(self):
        return f"Task at ({self.x}, {self.y})"

    def __repr__(self):
        return f"Task at ({self.x}, {self.y}) with reward {self.reward}"
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
# Robot class with random position and speed initialization
class Robot:
    def __init__(self, grid_size):
        # Random x, y position within grid boundaries
        self.x = np.random.randint(0, grid_size[0])
        self.y = np.random.randint(0, grid_size[1])
        # Random speed between 0.5 and 1.0
        self.speed = np.random.uniform(0.5, 1.0)

    def __repr__(self):
        return f"Robot at ({self.x}, {self.y}) with speed {self.speed:.2f}"
    

def init_robots(num_robots, grid_size):
    robots = [Robot(grid_size) for _ in range(num_robots)]
    return robots

def init_tasks(num_tasks, grid_size):
    tasks = [Task(
        np.random.randint(0, grid_size[0]),  # random x position
        np.random.randint(0, grid_size[1]))  # random y position
        for _ in range(num_tasks)]
    return tasks

def get_rewards(tasks):
    return [task.reward for task in tasks]

def visualize(robots, tasks, grid_size):
    # TASKS VIS
    reward_grid = np.zeros(grid_size)
    for task in tasks:
        reward_grid[task.x, task.y] = task.reward

    mask = reward_grid == 0

    # Plot the heatmap for tasks with rewards
    plt.figure(figsize=(10, 8))
    sns.heatmap(reward_grid, mask=mask, annot=True, fmt=".0f", cmap="coolwarm", 
                linewidths=0.5, square=True, cbar_kws={"label": "Reward"}, 
                annot_kws={"color": "black"})
    
    # ROBOT VIS
    for robot in robots:
        plt.scatter(robot.y + 0.5, robot.x + 0.5, color='lime', s=200, marker='X', edgecolors='black', label="Robot")

    # To avoid clutter, ensure we only add the legend once
    handles, labels = plt.gca().get_legend_handles_labels()
    if len(handles) == 0:
        plt.legend(["Robot"], loc='upper right')

    plt.title("Task Rewards Heatmap with Robot Positions")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.show()

def create_robot_task_matrix(robots, tasks):
    r = len(robots)
    t = len(tasks)
    
    # Initialize the r x t x 2 matrix
    matrix = np.zeros((r, t, 2))
    
    for i, robot in enumerate(robots):
        for j, task in enumerate(tasks):
            # Calculate Euclidean distance
            distance = np.sqrt((robot.x - task.x)**2 + (robot.y - task.y)**2)
            
            # Populate the matrix with (speed, distance)
            matrix[i, j] = [robot.speed, distance]
    
    return matrix

def get_allocations(ExBids, ExCare):
# ExCare = [[5, 6, 7, 8]]


# ExBids = [[5, 6, 7, 8],
#           [2, 6, 3, 4],
#           [6, 4, 2, 3],
#           [8, 5, 1, 2]]

    output = [[0 for _ in range(len(ExBids[0]))] for _ in range(len(ExBids))]

    used_columns = set()

    # Run a for loop for the number of rows in ExBids
    for _ in range(len(ExBids)):
        while True:  # Repeat the loop until a valid max value is found
            max_value = -float('inf')  # Reset max value for each iteration
            max_row, max_col = 0, 0  # Reset indices for each iteration

            # Loop through the grid to find the maximum value and its indices
            for row in range(len(ExBids)):
                for col in range(len(ExBids[row])):
                    if ExBids[row][col] > max_value:
                        max_value = ExBids[row][col]
                        max_row, max_col = row, col

            # If this column has already been used for another max, invalidate and try again
            if max_col in used_columns:
                ExBids[max_row][max_col] = -float('inf')
                continue  # Retry to find another maximum value in a different column

            # Store this column as used
            used_columns.add(max_col)

            # Extract the column where the max value is located
            column_values = [ExBids[row][max_col] for row in range(len(ExBids))]

            # Remove the max value from the column and find the second highest value
            column_values.remove(max_value)
            second_highest_value = max(column_values)

            # Find the corresponding value from ExCare
            corresponding_care_value = ExCare[max_col]

            # Check if second_highest_value is -inf, if so, use max_value instead
            if second_highest_value == -float('inf'):
                difference = corresponding_care_value - max_value
            else:
                # Calculate the difference with second highest value
                difference = corresponding_care_value - second_highest_value

            # Store the difference at the (max_row, max_col) index of the output grid
            output[max_row][max_col] = difference

            # Modify the entire row of max_row in ExBids to -float('inf')
            ExBids[max_row] = [-float('inf')] * len(ExBids[max_row])

            # # Modify the corresponding ExCare index (max_col) to -float('inf')
            # ExCare[max_col] = -float('inf')

            # # Print the current state of ExBids, ExCare, and output after each iteration
            # print(f"After iteration {_ + 1}:")
            # print("Output grid:")
            # for row in output:
            #     print(row)

            # print("Modified ExBids grid:")
            # for row in ExBids:
            #     print(row)

            # print("Modified ExCare list:")
            # print(ExCare)
            # print("\n----------------------\n")

            break  # Exit the while loop after a valid max value is processed
    output_dict = {}
    for row_idx, row in enumerate(output):
        for col_idx, value in enumerate(row):
            if value != 0:
                output_dict[row_idx] = (col_idx, value)
                break  # Since only one non-zero value is expected per row

    # print("Output dictionary:")
    # print(output_dict)
    return output_dict

