# TODO: renformat the imports for inginious
from agent import AlphaBetaAgent
import minimax as minimax
import pontu_state as pontu_state
import numpy as np


class MyAgent(AlphaBetaAgent):

	"""
	This is the skeleton of an agent to play the Tak game.
	"""
	def get_action(self, state, last_action, time_left):
		self.last_action = last_action
		self.time_left = time_left
		return minimax.search(state, self)

	"""
	The successors function must return (or yield) a list of
	pairs (a, s) in which a is the action played to reach the
	state s.
	"""
	def successors(self, state: pontu_state.PontuState):
		actions = self.get_simply_actions(state)
		# actions = state.get_current_player_actions()

		np.random.shuffle(actions)
		for action in actions:
			new_state = state.copy()
			new_state.apply_action(action)	
			yield (action, new_state)

	def get_simply_actions(self, state: pontu_state.PontuState):
		available_actions_3 = [i for i in range(0, state.size - 1)]
		available_actions_4 = [i for i in range(0, state.size)]
		actions = []
		
		for i in range(state.size-2): # for each pawn
			if not state.blocked[state.cur_player][i]: # if the pawn is not blocked
				dirs = state.move_dir(state.cur_player,i)
				for dir in dirs: # for each direction the pawn can move towards
					for pawn in range(state.size-2):
						pos = state.get_pawn_position(1- self.id,pawn)
						if (pos[0] - 1) in available_actions_3 and (pos[1] in available_actions_4):
							if state.h_bridges[pos[1]][pos[0] - 1]:
								actions.append((i,dir, 'h', pos[0] - 1, pos[1]))
						if pos[0] in available_actions_3 and pos[1] in available_actions_4:
							if state.h_bridges[pos[1]][pos[0]]:
								actions.append((i,dir, 'h', pos[0], pos[1]))
						if (pos[1] - 1) in available_actions_3 and pos[0] in available_actions_4:
							if state.v_bridges[pos[1]-1][pos[0]]: 
								actions.append((i,dir, 'v', pos[0], pos[1] - 1))
						if pos[1] in available_actions_3 and pos[0] in available_actions_4:
							if state.v_bridges[pos[1]][pos[0]]:
								actions.append((i,dir, 'v', pos[0], pos[1]))
		
		# then the valid actions consist only on the removal of one bridge
		if len(actions) == 0:
			for pawn in range(state.size-2):
				pos = state.get_pawn_position(1- self.id,pawn)
				if (pos[0] - 1) in available_actions_3 and (pos[1] in available_actions_4):
					if state.h_bridges[pos[1]][pos[0] - 1]:
						actions.append((None, None, 'h', pos[0] - 1, pos[1]))
				if pos[0] in available_actions_3 and pos[1] in available_actions_4:
					if state.h_bridges[pos[1]][pos[0]]:
						actions.append((None, None, 'h', pos[0], pos[1]))
				if (pos[1] - 1) in available_actions_3 and pos[0] in available_actions_4:
					if state.v_bridges[pos[1]-1][pos[0]]:
						actions.append((None, None, 'v', pos[0], pos[1] - 1))
				if pos[1] in available_actions_3 and pos[0] in available_actions_4:
					if state.v_bridges[pos[1]][pos[0]]:
						actions.append((None, None, 'v', pos[0], pos[1]))
		return actions

	"""
	The cutoff function returns true if the alpha-beta/minimax
	search has to stop and false otherwise.
	"""
	def cutoff(self, state, depth):
		if state.game_over():
			return True
		if depth > 2:
			return True
		return False

	"""
	The evaluate function must return an integer value
	representing the utility function of the board.
	"""
	def evaluate(self, state: pontu_state.PontuState):
		nb_bridges_opponent = 0
		nb_bridges_me = 0

		# Bridge count
		for i in range(0, state.size - 2):
			# Count opponent bridges
			for bridge in state.adj_bridges(1 - self.id, i).values():
				if bridge:
					nb_bridges_opponent += 1
			
			# Count my bridges
			for bridge in state.adj_bridges(self.id, i).values():
				if bridge:
					nb_bridges_me += 1

		# Save one of our spawn (only if the next case has two or more bridges)
		nb_pawn_safe_me = 0
		for i in range(0, state.size - 2):
			if sum(1 for v in state.adj_bridges(self.id, i).values() if v == True) >= 2:
				nb_pawn_safe_me += 1

		# Block opponent spawn
		nb_pawn_blocked_opponent = 0
		for i in range(0, state.size - 2):
			if state.is_pawn_blocked(1 - self.id, i):
				nb_pawn_blocked_opponent += 1
		
		# Perfer dispersive play
		list = []
		for i in range(0, state.size - 2):
			list.append(state.get_pawn_position(self.id, i))

		counter = 0
		for i in range(0, len(list)):
			for j in range(0, len(list)):
				if list[i] != list[j] and ((list[i][1] - list[j][1]) + (list[i][0] - list[j][0])) == 1:
					counter += 1

		# Set a weight to our bridges to make playing offensive more interesting
		# AABBCC.DD
		#   - AA: number of oppenent pawn blocked
		#   - BB: number of my pawn safe
		#   - CC: number of oppenent's bridges
		#   - DD: number of my bridges
		opponents_bridge = ((state.size - 2) * 4 - nb_bridges_opponent)
		my_bridge = nb_bridges_me
		my_pawn_safe = nb_pawn_safe_me
		opponents_pawn_blocked = nb_pawn_blocked_opponent
		dispersive_play = ((state.size - 2)**2 - counter)

		opponents_bridge_weight = 1
		my_bridge_weight = 1e-4
		my_pawn_safe_weight = 1e-2
		opponents_pawn_blocked_weight = 1e4
		dispersive_play_weight = 0

		weight = opponents_bridge * opponents_bridge_weight + \
		        my_bridge * my_bridge_weight + \
		        my_pawn_safe * my_pawn_safe_weight + \
				opponents_pawn_blocked * opponents_pawn_blocked_weight + \
				dispersive_play * dispersive_play_weight
		
		# weight = ((state.size - 2) * 4 - nb_bridges_opponent) + nb_bridges_me * 1e-4 \
		# + nb_pawn_safe_me * 1e-2 + nb_pawn_blocked_opponent * 1e4 + ((state.size - 2)**2 - counter) * 1e-6
		
		# print("", weight, end="- ")
		return weight
		
	def get_name(self):
		return "better agent 2"