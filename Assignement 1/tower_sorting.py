#!/usr/bin/env python
"""
Name of the author(s):
- Auguste Burlats <auguste.burlats@uclouvain.be>
"""
import time
import sys
from copy import deepcopy
from search import *


#################
# Problem class #
#################
class TowerSorting(Problem):

    def actions(self, state):
        # An action is a tuple (list index src, list index dest)
        for i in range(state.number):
            # No number in this tower
            if len(state.grid[i]) == 0: continue #  or (len(state.grid[i]) == state.size and len(np.unique(state.grid[i])) == 1)
            # Move each top number to all other tower
            for j in range(state.number):
                if len(state.grid[j]) != state.size and (i != j):
                    yield (i, j)

    def result(self, state, action):
        new_grid = deepcopy(state.grid)
        top = new_grid[action[0]].pop()
        new_grid[action[1]].append(top)
        return State(state.number, state.size, new_grid, "tower " + str(action[0]) + " -> tower " + str(action[1]))

    def goal_test(self, state):
        is_goal = True
        for i in range(state.number):
            if not is_goal or (len(state.grid[i]) != 0 and len(state.grid[i]) != state.size):
                is_goal = False
                break
            current = state.grid[i][0] if len(state.grid[i]) > 0 else None
            for j in range(len(state.grid[i])):
                if state.grid[i][j] != current: 
                    is_goal = False
                    break
        return is_goal 

###############
# State class #
###############
class State:

    def __init__(self, number, size, grid, move="Init", depth=0):
        self.number = number    # Number of towers
        self.size = size        # Size of each tower 
        self.grid = grid        # Grid of the towers
        self.move = move        # Move that led to this state

    def __str__(self):
        s = self.move + "\n"
        for i in reversed(range(self.size)):
            for tower in self.grid:
                if len(tower) > i:
                    s += "".join(tower[i]) + " "
                else:
                    s += ". "
            s += "\n"
        return s

    def __eq__(self, other):
        return hash(self) == hash(other)
    
    def __hash__(self):
        tupled_grid = tuple([tuple(tower) for tower in self.grid])
        return hash((self.number, self.size, tupled_grid))


######################
# Auxiliary function #
######################
def read_instance_file(filepath):
    with open(filepath) as fd:
        lines = fd.read().splitlines()

    number_tower, size_tower = tuple([int(i) for i in lines[0].split(" ")])
    initial_grid = [[] for i in range(number_tower)]
    for row in lines[1:size_tower+1]:
        elems = row.split(" ")
        for index in range(number_tower):
            if elems[index] != '.':
                initial_grid[index].append(elems[index])

    for tower in initial_grid:
        tower.reverse()

    return number_tower, size_tower, initial_grid


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: ./sort_tower.py <path_to_instance_file>")
    filepath = sys.argv[1]

    number, size, initial_grid = read_instance_file(filepath)

    init_state = State(number, size, initial_grid, "Init")
    problem = TowerSorting(init_state)

    # Example of search
    start_timer = time.perf_counter()
    node, nb_explored, remaining_nodes = breadth_first_graph_search(problem) # BFS find the shortest path
    end_timer = time.perf_counter()

    # Example of print
    path = node.path()
    
    for n in path:
        print(n.state)

    print("* Execution time:\t", str(end_timer - start_timer))
    print("* Path cost to goal:\t", node.depth, "moves")
    print("* #Nodes explored:\t", nb_explored)
    print("* Queue size at goal:\t",  remaining_nodes)
