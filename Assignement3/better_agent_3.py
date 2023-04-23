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
			
	"""
	Return a list of action that sould be considered by the agent
	"""
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
		a = 0
		for i in range(0, state.size - 2):
			if state.is_pawn_blocked(self.id, i):
				a += 1
			if state.is_pawn_blocked(1 - self.id, i):
				a += 1
		if depth > a + 1:
			return True
		return False

	"""
	The evaluate function must return an integer value
	representing the utility function of the board.
	"""
	def evaluate(self, state: pontu_state.PontuState):
		# Priority 1: eleminate opponent's pawn
		# Priority 2: save own pawn
		# Priority 3: remove opponent's bridge
		# Priority 4: minimize our bridge loss

		# Adapt the weight of each priority according to the game state
		attack_adaptive = 2
		defense_adaptive = 1
		for paw in range(state.size - 2):
			if state.is_pawn_blocked(1 - self.id, paw):
				attack_adaptive += 1
			if state.is_pawn_blocked(self.id, paw):
				defense_adaptive += 1
				if defense_adaptive > 2:
					attack_adaptive -= 1
				
		# Get the result weight of this action
		missing_brides_op = 0
		missing_brides_us = 0
		for pawn in range(state.size - 2):
			nb_miss_op = 0
			nb_miss_us = 0
			for dir in ['EAST', 'NORTH', 'WEST', 'SOUTH']:
				# Opponent's pawns
				if state.is_pawn_blocked(1 - self.id, pawn) or not state.adj_bridges(1 - self.id, pawn)[dir]:
					nb_miss_op += 2
				elif state.adj_pawns(1 - self.id, pawn)[dir]:
					nb_miss_op += 1
				
				# Our pawns
				if state.is_pawn_blocked(self.id, pawn) or not state.adj_bridges(self.id, pawn)[dir]:
					nb_miss_us += 2
				elif state.adj_pawns(self.id, pawn)[dir]:
					nb_miss_us += 1

				missing_brides_op += nb_miss_op ** 2
				missing_brides_us += nb_miss_us ** 2

		res = missing_brides_op ** attack_adaptive - missing_brides_us ** defense_adaptive
		return res

		
	def get_name(self):
		return "better agent 3"