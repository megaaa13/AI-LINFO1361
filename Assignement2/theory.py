import numpy as np

# Question 1.3 ==> generate h(n)
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

# Create f(n) = g(n) + h(n)
g = [[1, 's', 'x', 'x',  9,   10,  11],
     [2, 'x', 'x',  7,   8,   'x', 12],
     [3,  4,   5,   6,  'x',  'x', 13],
     [4,  5,   6,  'x',  16,  15,  14],
     [5,  6,  'x', 'g',  17,  'x', 15],
     [6,  7,   8,  'x',  'x', 13,  14],
     [7,  8,   9,  10,   11,  12,  13]]

h = [[7, 's', 'x', 'x',  5,   6,  7],
     [6, 'x', 'x',  3,   4,  'x', 6],
     [5,  4,   3,   2,  'x', 'x', 5],
     [4,  3,   2,  'x',  2,   3,  4],
     [3,  2,  'x', 'g',  1,  'x', 3],
     [4,  3,   2,  'x', 'x',  3,  4],
     [5,  4,   3,   2,   3,   4,  5]]

# Sum the two matrices
f = []
for i in range(7):
    l = []
    for j in range(7):
        if g[i][j] == 'x':
            l.append('x')
        elif g[i][j] == 's':
            l.append('s')
        elif g[i][j] == 'g':
            l.append('g')
        else:
            l.append(g[i][j] + h[i][j])
    f.append(l)
    

# Print the matrix
for i in range(7):
    for j in range(7):
        print(f[i][j], end=' ')
    print()


# Print the LaTeX code
for i in range(7):
    for j in range(7):
        print("\\node at ({}.5, {}.5) {{{}}};".format(i, j, int(maze[i, j])))