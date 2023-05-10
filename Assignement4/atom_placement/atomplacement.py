#! /usr/bin/env python3
"""NAMES OF THE AUTHOR(S): Auguste Burlats <auguste.burlats@uclouvain.be>"""
from search import *

class State:

    def __init__(self, n_sites, n_types, edges, energy_matrix, sites=None):
        self.k = len(n_types)
        self.n_types = n_types
        self.n_sites = n_sites
        self.n_edges = len(edges)
        self.edges = edges
        self.energy_matrix = energy_matrix
        if sites is None:
            self.sites = self.build_init()
        else:
            self.sites = sites

    # an init state building is provided here but you can change it at will
    def build_init(self):
        sites = []
        for atom_type, quantity in enumerate(self.n_types):
            for i in range(quantity):
                sites.append(atom_type)

        return sites

    def __str__(self):
        s = ''
        for v in self.sites:
            s += ' ' + str(v)
        return s

class AtomPlacement(Problem):

    # if you want you can implement this method and use it in the maxvalue and randomized_maxvalue functions
    def successor(self, state: State):
        l = []
        for edge in state.edges:
            sites = state.sites.copy()
            #swap
            sites[edge[0]], sites[edge[1]] = sites[edge[1]], sites[edge[0]]
            l.append((state, State(state.n_sites, state.n_types, state.edges, state.energy_matrix, sites)))
        return l

    # if you want you can implement this method and use it in the maxvalue and randomized_maxvalue functions
    def value(self, state: State):
        return sum([state.energy_matrix[state.sites[i]][state.sites[j]] for i, j in state.edges])


def read_instance(instanceFile):
    file = open(instanceFile)
    line = file.readline()
    n_sites = int(line.split(' ')[0])
    k = int(line.split(' ')[1])
    n_edges = int(line.split(' ')[2])
    edges = []
    file.readline()

    n_types = [int(val) for val in file.readline().split(' ')]
    if sum(n_types) != n_sites:
        print('Invalid instance, wrong number of sites')
    file.readline()

    energy_matrix = []
    for i in range(k):
        energy_matrix.append([int(val) for val in file.readline().split(' ')])
    file.readline()

    for i in range(n_edges):
        edges.append([int(val) for val in file.readline().split(' ')])

    return n_sites, n_types, edges, energy_matrix


# Attention : Depending of the objective function you use, your goal can be to maximize or to minimize it
#? 8 on 14 on inginious (12 required)
def maxvalue(problem :Problem, limit=100, callback=None):
    current = LSNode(problem, problem.initial, 0)
    best = current
    for i in range(limit):
        current = min(best.expand(), key=lambda x: x.value())
        if best.value() > current.value():
            best = current
    return best


# Attention : Depending of the objective function you use, your goal can be to maximize or to minimize it
#! Currently not working well
def randomized_maxvalue(problem, limit=100, callback=None):
    current = LSNode(problem, problem.initial, 0)
    for _ in range(limit):
        list = [(child, child.value()) for child in current.expand()]
        list.sort(key=lambda x: x[1])
        current = list[random.randint(0, 4)][0]
    return current


#####################
#       Launch      #
#####################
if __name__ == '__main__':
    info = read_instance(sys.argv[1])
    init_state = State(info[0], info[1], info[2], info[3])
    ap_problem = AtomPlacement(init_state)
    step_limit = 100
    node = randomized_maxvalue(ap_problem, 100, step_limit)
    state = node.state
    print(state)