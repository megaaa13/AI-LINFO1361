from agent import AlphaBetaAgent
import minimax
import pontu_state as pontu_state


class MyAgent(AlphaBetaAgent):

	def get_action(self, state, last_action, time_left):
		self.last_action = last_action
		self.time_left = time_left
		return minimax.search(state, self)

	def successors(self, state: pontu_state.PontuState):
		'''
		The successors function must return (or yield) a list of pairs (a, s) 
		in which a is the action played to reach the state s.
		'''
		# Get a list of simply actions
		actions = self.get_simply_actions(state)

		# Compute the next state for each action
		states = [None] * len(actions)  # pre-allocate list
		for i, action in enumerate(actions):
			new_state = state.copy()
			new_state.apply_action(action)
			states[i] = (action, new_state)
		
		# Sort the states according to the evaluation function
		if (self.id == state.cur_player):
			states.sort(key=lambda x: self.evaluate(x[1]), reverse=True)
		else:
			states.sort(key=lambda x: self.evaluate(x[1]))
		
		# Return the sorted list of states
		return states

	def cutoff(self, state, depth):
		'''
		The cutoff function returns true if the alpha-beta/minimax search has to stop and false otherwise.
		'''
		# Game over
		if state.game_over():
			return True

		# Depth limit
		initial_depth = 2
		adaptive_depth = 0
		
		# Adaptive depth
		for i in range(0, state.size - 2):
			if state.is_pawn_blocked(self.id, i):
				adaptive_depth += 1
			if state.is_pawn_blocked(1 - self.id, i):
				adaptive_depth += 1
			
		# Check depth
		if depth > initial_depth + adaptive_depth:
			return True
		return False

	def evaluate(self, state: pontu_state.PontuState):
		'''
		The evaluate function must return an integer value representing the utility function of the board.
		'''
		# Attack and defense adaptive weights
		priority_attack = 1
		priority_defense = 1
		for paw in range(state.size - 2):
			if state.is_pawn_blocked(1 - self.id, paw):
				priority_attack += 1
			if state.is_pawn_blocked(self.id, paw):
				priority_defense += 1

		# Evaluate parameters
		nb_bridges_opponent = 0
		nb_bridges_me = 0

		nb_pawn_blocked_opponent = 0
		nb_pawn_blocked_me = 0

		nb_pawn_safe_me = 0
		nb_pawn_safe_opponent = 0

		nb_near_center_me = 0
		nb_near_center_opponent = 0

		nb_available_moves_me = 0
		nb_available_moves_opponent = 0	

		# Parameters calculation
		my_pawns_pos = []
		opponent_pawns_pos = []
		for k in range(0, state.size - 2):
			my_pawns_pos.append(state.get_pawn_position(self.id, k))
			opponent_pawns_pos.append(state.get_pawn_position(1 - self.id, k))

		for i in range(0, state.size - 2):
			# Briges
			for bridge in state.adj_bridges(1 - self.id, i).values():
				if bridge: nb_bridges_opponent += 1
			for bridge in state.adj_bridges(self.id, i).values():
				if bridge: nb_bridges_me += 1

			# Blocked pawns
			if state.is_pawn_blocked(1 - self.id, i):
				nb_pawn_blocked_opponent += 1
			if state.is_pawn_blocked(self.id, i):
				nb_pawn_blocked_me += 1

			# Is pawn in a safe position
			if sum(1 for v in state.adj_bridges(self.id, i).values() if v == True) >= 2:
				nb_pawn_safe_me += 1
			if sum(1 for v in state.adj_bridges(1 - self.id, i).values() if v == True) >= 2:
				nb_pawn_safe_opponent += 1
			
			# Aim the center by manhattan distance
			nb_near_center_me += self.manhattan_distance_to_center(state.get_pawn_position(self.id, i), state.size) if not state.is_pawn_blocked(self.id, i) else 0
			nb_near_center_opponent += self.manhattan_distance_to_center(state.get_pawn_position(1 - self.id, i), state.size)if not state.is_pawn_blocked(1 - self.id, i) else 0

			# Nb available moves
			nb_available_moves_me += len(state.move_dir(self.id, i))
			nb_available_moves_opponent += len(state.move_dir(1 - self.id, i))

		# Attack and defense weights
		bridges, w1			= priority_defense * nb_bridges_me  		  - priority_attack  * nb_bridges_opponent,			3
		pawns, w2			= priority_attack  * nb_pawn_blocked_opponent - priority_defense * nb_pawn_blocked_me,		    1e3
		safe_pawns, w3 		= priority_defense * nb_pawn_safe_me 		  - priority_attack  * nb_pawn_safe_opponent, 		1
		near_center, w4 	= priority_attack  * nb_near_center_opponent  - priority_defense * nb_near_center_me,			1e-2
		available_moves, w5 = priority_defense * nb_available_moves_me 	  - priority_attack  * nb_available_moves_opponent, 2

		# Return
		return bridges * w1 + pawns * w2 + safe_pawns * w3 + near_center * w4 + available_moves * w5 

	#############################################################################################################
	#											Helper functions												#
	#############################################################################################################
	def get_simply_actions(self, state: pontu_state.PontuState):
		''''
		Return a list of actions that can be played by the current player.
		This actions are, for each pawn, the possible directions it can take.
		And the bridges to removes are only bridges adjacent to the opponent player's pawns.
		'''

		# Create number list for lenght of the board
		available_actions_3 = [i for i in range(0, state.size - 1)]
		available_actions_4 = [i for i in range(0, state.size)]

		# Create a list of actions
		actions = []
		for i in range(state.size-2):  # for each pawn
			if not state.blocked[state.cur_player][i]:  # if the pawn is not blocked
				dirs = state.move_dir(state.cur_player, i)
				for dir in dirs:  # for each direction the pawn can move towards
					for pawn in range(state.size-2):
						pos = state.get_pawn_position(1 - state.cur_player, pawn)
						if (pos[0] - 1) in available_actions_3 and (pos[1] in available_actions_4):
							if state.h_bridges[pos[1]][pos[0] - 1]:
								actions.append((i, dir, 'h', pos[0] - 1, pos[1]))
						if pos[0] in available_actions_3 and pos[1] in available_actions_4:
							if state.h_bridges[pos[1]][pos[0]]:
								actions.append((i, dir, 'h', pos[0], pos[1]))
						if (pos[1] - 1) in available_actions_3 and pos[0] in available_actions_4:
							if state.v_bridges[pos[1]-1][pos[0]]:
								actions.append((i, dir, 'v', pos[0], pos[1] - 1))
						if pos[1] in available_actions_3 and pos[0] in available_actions_4:
							if state.v_bridges[pos[1]][pos[0]]:
								actions.append((i, dir, 'v', pos[0], pos[1]))
		
		# If no points can be moved
		if len(actions) == 0:
			for pawn in range(state.size-2):
				pos = state.get_pawn_position(1 - state.cur_player, pawn)
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
	
	def manhattan_distance_to_center(self, pos, state_size):
		'''
		Return the manhattan distance between two positions
		'''
		center = state_size // 2 + 1
		return (abs(center - pos[0]) + abs(center - pos[1])) ** 2