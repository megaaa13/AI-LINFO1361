import time
from search import *

#################
# Problem class #
#################

class SoftFlow(Problem):

    def __init__(self, initial, goal=None):
        self.initial = initial
        self.goal = goal if goal else self.get_goal(self.initial)
        
    def actions(self, state):
        # An action is a tuple (char, (xTo, yTo, integerTo))
        actions = []

        # From the enc of the cable, we can go in 4 directions
        # only if the cell is empty
        for char in state.positions.keys():
            if not state.goals_reached[int(state.positions[char][2])]:
                for i in (-1, 0, 1): 
                    for j in (-1, 0, 1):
                        # Not diagonal or the same cell AND the goal cell is empty
                        if i != j and i != -j and state.grid[state.positions[char][0] + i][state.positions[char][1] + j] == ' ':
                            actions.append((char, (state.positions[char][0] + i, state.positions[char][1] + j, state.positions[char][2])))
        return actions

    def result(self, state, action):
        # Copy the grid
        new_grid = list(state.grid)
        # Replace the char in the grid by the number
        new_grid[state.positions[action[0]][0]] = new_grid[state.positions[action[0]][0]][0:state.positions[action[0]][1]] + tuple(state.positions[action[0]][2]) + new_grid[state.positions[action[0]][0]][state.positions[action[0]][1] + 1:]
        # Place the char in the new position
        new_grid[action[1][0]] = new_grid[action[1][0]][0:action[1][1]] + tuple(action[0]) + new_grid[action[1][0]][action[1][1] + 1:]
        
        # Create the new state
        new_state = State(tuple(new_grid), state.positions, state.goals_reached)
        new_state.positions[action[0]] = action[1]
        return new_state

    def goal_test(self, state):
        # Check if positions are nearby the goal
        for char in state.positions.keys():
            if not state.goals_reached[int(state.positions[char][2])] and (abs(state.positions[char][0] - self.goal[char][0]) + abs(state.positions[char][1] - self.goal[char][1]) == 1):
                # If the goal is reached, we update the state
                state.goals_reached[int(state.positions[char][2])] = True
                # Apply the move to the grid
                new_grid = list(state.grid)
                new_grid[state.positions[char][0]] = new_grid[state.positions[char][0]][0:state.positions[char][1]] + tuple(state.positions[char][2]) + new_grid[state.positions[char][0]][state.positions[char][1] + 1:]
                state.grid = tuple(new_grid)
        
        # Check if all goals are reached
        return all(state.goals_reached)

    def h(self, node):
        h = 0.0

        # Manhattan distance formula
        d = lambda x1, y1, x2, y2: abs(x1 - x2) + abs(y1 - y2)

        # Compute heuristic value
        for char in node.state.positions.keys():
            if not node.state.goals_reached[int(node.state.positions[char][2])] :
                h += d(node.state.positions[char][0], node.state.positions[char][1], self.goal[char][0], self.goal[char][1])

        # Add preference for already good path
        for i in node.state.goals_reached:
            if i == False:
                h += 100

        # Add dependence to depth
        h += node.depth

        return h
        

    def load(path):
        with open(path, 'r') as f:
            lines = f.readlines()
            
        state = State.from_string(''.join(lines))
        return SoftFlow(state)
    
    def __make_alphabet(self):
        # Return the alphabet-number links : {'a': 0, 'b': 1, ...}
        return {str(idx): char for idx, char in enumerate('abcdefghij')}
    
    def get_goal(self, state):
        alphabet = self.__make_alphabet()

        # Return the goal in the grid
        goal = {}
        for i, row in enumerate(state.grid):
            for j, char in enumerate(row):
                if char.isdigit():
                    goal[alphabet[char]] = (i, j)
        return goal



###############
# State class #
###############

class State:

    def __init__(self, grid, positions=None, goals_reached=None):
        # Grid values
        self.nbr = len(grid)     # number of rows
        self.nbc = len(grid[0])  # number of columns
        self.grid = grid

        # Fast access
        self.hash = hash(self.grid)
        self.positions = positions.copy() if positions else self.get_positions(self)  # positions of the end of cable in the grid
        self.goals_reached = goals_reached.copy() if goals_reached else [False for _ in range(len(self.positions))]  # goals_reached
        
    def __str__(self):
        return '\n'.join(''.join(row) for row in self.grid)

    def __eq__(self, other_state):
        return self.hash == other_state.hash

    def __hash__(self):
        return hash(self.grid)
    
    def __lt__(self, other):
        return self.hash < other.hash

    def from_string(string):
        lines = string.strip().splitlines()
        return State(tuple(
            map(lambda x: tuple(x.strip()), lines)
        ))
    
    def __make_alphabet(self):
        # Return the alphabet-number links : {'a': 0, 'b': 1, ...}
        return {char: idx for idx, char in enumerate('abcdefghij')}
    
    def get_positions(self, state):
        alphabet = self.__make_alphabet()

        # Get the positions of the letters in the grid : {'a': (x1, y1, '0'), 'b': (x2, y2, '1'), ...}
        # if a is at position (x1, y1) and b is at position (x2, y2) in the grid
        positions = {}
        for i, row in enumerate(state.grid):
            for j, char in enumerate(row):
                idx = alphabet.get(char, None)
                if idx is not None:
                    positions[char] = (i, j, str(idx))
        return positions    
    


#####################
# Launch the search #
#####################

problem = SoftFlow.load(sys.argv[1])

start_timer = time.perf_counter()
node = astar_search(problem)
end_timer = time.perf_counter()

# example of print
path = node.path()

#print('Number of moves: ', str(node.depth))
#for n in path:
#    print(n.state)  # assuming that the _str_ function of state outputs the correct format
#    print()

print("* Execution time:\t", str(end_timer - start_timer))
