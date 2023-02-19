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

    # TODO : Faire un itérateur et non une liste
    def actions(self, state):
        actions = [] # Tuples containing top of the tower to
        for i in range(state.number):
            if len(state.grid[i]) == 0:
                continue
            for j in range(0, len(state.grid)):
                if (i == j):
                    continue
                if len(state.grid[j]) != state.size:
                    actions.append((i, j))
        return actions

    def result(self, state, action):
        new_grid = deepcopy(state.grid)
        top = new_grid[action[0]].pop()
        new_grid[action[1]].append(top)
        return State(state.number, state.size, new_grid, "tower " + str(action[0]) + " -> tower " + str(action[1]))

    def goal_test(self, state):
        if self.goal == state:
            return True       


###############
# State class #
###############
class State:

    def __init__(self, number, size, grid, move="Init"):
        self.number = number
        self.size = size
        self.grid = grid
        self.move = move

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
    goal = State(number, size, [[str(i)] * size for i in range(1, number)] + [[]], "Goal") # Goal state
    problem = TowerSorting(init_state, goal)
    # Example of search
    start_timer = time.perf_counter()
    node, nb_explored, remaining_nodes = depth_first_graph_search(problem)
    end_timer = time.perf_counter()

    # Example of print
    path = node.path()

    for n in path:
        # assuming that the __str__ function of state outputs the correct format
        print(n.state)

    print("* Execution time:\t", str(end_timer - start_timer))
    print("* Path cost to goal:\t", node.depth, "moves")
    print("* #Nodes explored:\t", nb_explored)
    print("* Queue size at goal:\t",  remaining_nodes)
