from search import *

#################
# Problem class #
#################

class SoftFlow(Problem):

    def __init__(self, initial, goal=None):
        self.initial = initial
        self.goal = goal if goal else self.get_goal(self.initial)
        self.state = 1
        
        
    def actions(self, state):
        for k in state.positions.keys():
            if state.goal_reached[int(state.positions[k][2])] == True:
                continue
            for i in (-1, 0, 1):
                for j in (-1, 0, 1):
                    if i != j and i != -j and state.grid[state.positions[k][0] + i][state.positions[k][1] + j] == ' ':
                        yield (k, (state.positions[k][0] + i, state.positions[k][1] + j, state.positions[k][2]))

    def result(self, state, action):
        #print(action)
        new_grid = list(state.grid)
        # replacing the letter by the pipe
        new_grid[state.positions[action[0]][0]] = new_grid[state.positions[action[0]][0]][0:state.positions[action[0]][1]] + tuple(state.positions[action[0]][2]) + new_grid[state.positions[action[0]][0]][state.positions[action[0]][1] + 1:]
        # placing the letter
        new_grid[action[1][0]] = new_grid[action[1][0]][0:action[1][1]] + tuple(action[0]) + new_grid[action[1][0]][action[1][1] + 1:]
        #new_grid[state.positions[action[0]][0]] = list(new_grid[state.positions[action[0]][0]])
        #new_grid[state.positions[action[0]][0]][state.positions[action[0]][1]] = state.positions[action[0]][2]
        #new_grid[state.positions[action[0]][0]] = tuple(new_grid[state.positions[action[0]][0]])
        #new_grid[action[1][0]] = list(new_grid[action[1][0]])
        #new_grid[action[1][0]][action[1][1]] = action[0]
        #new_grid[action[1][0]] = tuple(new_grid[action[1][0]])
        new_state = State(tuple(new_grid), state.positions, state.goal_reached)
        new_state.positions[action[0]] = action[1]
        #print(new_state)
        return new_state

    def goal_test(self, state):
        #check if positions are nearby the goal
        for k in state.positions.keys():
            if (abs(state.positions[k][0] - self.goal[k][0]) + abs(state.positions[k][1] - self.goal[k][1]) == 1) and state.goal_reached[int(state.positions[k][2])] == False:
                state.goal_reached[int(state.positions[k][2])] = True
                #print("Goal reached: ", k)
                #modify letters in the grid
                new_grid = list(state.grid)
                new_grid[state.positions[k][0]] = new_grid[state.positions[k][0]][0:state.positions[k][1]] + tuple(state.positions[k][2]) + new_grid[state.positions[k][0]][state.positions[k][1] + 1:]
                #new_grid[state.positions[k][0]] = list(new_grid[state.positions[k][0]])
                #new_grid[state.positions[k][0]][state.positions[k][1]] = state.positions[k][2]
                #new_grid[state.positions[k][0]] = tuple(new_grid[state.positions[k][0]])
                state.grid = tuple(new_grid)
        #check if all goals are reached
        for k in state.goal_reached:
            if k == False:
                return False
        #print("Goals: ", self.goal)
        return True

    def h(self, node):
        h = 0
        for k in node.state.positions.keys():
            if node.state.goal_reached[int(node.state.positions[k][2])] == False:
                h += abs(node.state.positions[k][0] - self.goal[k][0]) + abs(node.state.positions[k][1] - self.goal[k][1]) # manhattan distance
        #print(h)
        for i in node.state.goal_reached: # prefer to choose path with already good solutions
            if i == False:
                h += 100
        h += node.depth # optimal solution is the shortest one
        return h
        

    def load(path):
        with open(path, 'r') as f:
            lines = f.readlines()
            
        state = State.from_string(''.join(lines))
        #print(state)
        return SoftFlow(state)
    
    def get_goal(self, state):
        goal = {}
        for k in state.positions.keys():
            # search for the index in the grid 
            # and add it to the goal
            for i in range(state.nbr):
                for j in range(state.nbc):
                    if state.grid[i][j] == state.positions[k][2]:
                        goal[k[0]] = (i, j) # False means that the goal is not reached
        #print("Goals : ", goal)
        return goal


###############
# State class #
###############

class State:


    def __init__(self, grid, positions = None, goal = None):
        self.nbr = len(grid)
        self.nbc = len(grid[0])
        self.grid = grid
        self.positions = positions.copy() if positions else self.get_positions(self)
        self.goal_reached = goal.copy() if goal else [False for _ in range(len(self.positions))]
        self.hash = hash(self.grid)
        
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
    
    def get_positions(self, state):
        dict_alphabet = {}
        idx = 0
        for i in 'abcdefghijklmnopqrstuvwxyz':
            dict_alphabet[i] = idx
            idx += 1
        positions = {}
        idx = 0
        for i in range(state.nbr):
            for j in range(state.nbc):
                idx = dict_alphabet.get(state.grid[i][j], None)
                if idx != None:
                    positions[state.grid[i][j]] = (i, j, str(idx))
                    idx += 1
        #print("Positions : ", positions)
        return positions






#####################
# Launch the search #
#####################
#import time
problem = SoftFlow.load(sys.argv[1])
#start_timer = time.perf_counter()
node = astar_search(problem, display=False)
#end_timer = time.perf_counter()
# example of print
path = node.path()

print('Number of moves: ', str(node.depth))
for n in path:
    """if (n.depth == 0):
        print('initial state')
    elif (n.depth == node.depth):
        print('goal state')
    else:
        print('state', str(n.depth))"""
    print(n.state)  # assuming that the _str_ function of state outputs the correct format
    print()
#print("* Execution time:\t", str(end_timer - start_timer))
#print('Number of moves: ', str(node.depth))
