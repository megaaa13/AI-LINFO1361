import numpy as np

# Question 1.3
# Goal
gx = 3; gy = 2

# Manhattan distance and maze
d = lambda x, y: np.abs(x-gx) + np.abs(y-gy)
maze = np.zeros((7, 7))


# For each cell, compute the distance to the goal
for i in range(7):
    for j in range(7):
        maze[i, j] = d(i, j)

# Print the maze rotated 90 degrees
print(np.rot90(maze))

# Print the LaTeX code
for i in range(7):
    for j in range(7):
        print("\\node at ({}.5, {}.5) {{{}}};".format(i, j, int(maze[i, j])))