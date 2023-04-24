from agent import AlphaBetaAgent
import minimax
import pontu_state as pontu_state


class MyAgent(AlphaBetaAgent):

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
		states = []
		for action in actions:
			new_state = state.copy()
			new_state.apply_action(action)
			states.append((action, new_state))
		if (self.id == state.cur_player):
			states.sort(key=lambda x: self.evaluate(x[1]), reverse=True)
		else:
			states.sort(key=lambda x: self.evaluate(x[1]))
		return states

	def get_simply_actions(self, state: pontu_state.PontuState):
		available_actions_3 = [i for i in range(0, state.size - 1)]
		available_actions_4 = [i for i in range(0, state.size)]
		actions = []
		for i in range(state.size-2):  # for each pawn
			if not state.blocked[state.cur_player][i]:  # if the pawn is not blocked
				dirs = state.move_dir(state.cur_player, i)
				for dir in dirs:  # for each direction the pawn can move towards
					for pawn in range(state.size-2):
						pos = state.get_pawn_position(
							1 - state.cur_player, pawn)
						if (pos[0] - 1) in available_actions_3 and (pos[1] in available_actions_4):
							if state.h_bridges[pos[1]][pos[0] - 1]:
								actions.append(
									(i, dir, 'h', pos[0] - 1, pos[1]))
						if pos[0] in available_actions_3 and pos[1] in available_actions_4:
							if state.h_bridges[pos[1]][pos[0]]:
								actions.append((i, dir, 'h', pos[0], pos[1]))
						if (pos[1] - 1) in available_actions_3 and pos[0] in available_actions_4:
							if state.v_bridges[pos[1]-1][pos[0]]:
								actions.append(
									(i, dir, 'v', pos[0], pos[1] - 1))
						if pos[1] in available_actions_3 and pos[0] in available_actions_4:
							if state.v_bridges[pos[1]][pos[0]]:
								actions.append((i, dir, 'v', pos[0], pos[1]))

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
		if depth > 1 + a:
			return True
		return False

	"""
	The evaluate function must return an integer value
	representing the utility function of the board.
	"""
	
	def evaluate(self, state: pontu_state.PontuState):
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
		

		for i in range(0, state.size - 2):

			#Nb bridges for the opponent
			for bridge in state.adj_bridges(1 - self.id, i).values():
				if bridge:
					nb_bridges_opponent += 1
			#Nb bridges for me
			for bridge in state.adj_bridges(self.id, i).values():
				if bridge:
					nb_bridges_me += 1

			#Blocked pawn
			if state.is_pawn_blocked(1 - self.id, i):
				nb_pawn_blocked_opponent += 1

			if state.is_pawn_blocked(self.id, i):
				nb_pawn_blocked_me += 1

			#Is pawn in a safe position ?
			if sum(1 for v in state.adj_bridges(self.id, i).values() if v == True) >= 2:
				nb_pawn_safe_me += 1
			
			if sum(1 for v in state.adj_bridges(1 - self.id, i).values() if v == True) >= 2:
				nb_pawn_safe_opponent += 1
			
			# Aim the center
			nb_near_center_me += (state.size//2 + 1 - abs(state.get_pawn_position(self.id, i)[0]) + abs(state.size//2 + 1 - state.get_pawn_position(self.id, i)[1]))**2 if not state.is_pawn_blocked(self.id, i) else 0
			nb_near_center_opponent += (state.size//2 + 1 - abs(state.get_pawn_position(1 - self.id, i)[0]) + abs(state.size//2 + 1 - state.get_pawn_position(1 - self.id, i)[1]))**2 if not state.is_pawn_blocked(1 - self.id, i) else 0

			#Nb available moves
			nb_available_moves_me += len(state.move_dir(self.id, i))
			nb_available_moves_opponent += len(state.move_dir(1 - self.id, i))


		# Dispersed pawns (not too near from each other)
		dispersive = 0
		list = []
		for i in range(0, state.size - 2):
			list.append(state.get_pawn_position(self.id, i))

		for j in range(0, len(list)):
			if list[i] != list[j] and (abs(list[i][1] - list[j][1]) + abs(list[i][0] - list[j][0])) == 1:
				dispersive += 1

		for i in range(0, state.size - 2):
			list.append(state.get_pawn_position(1 - self.id, i))

		for j in range(0, len(list)):
			if list[i] != list[j] and (abs(list[i][1] - list[j][1]) + abs(list[i][0] - list[j][0])) == 1:
				dispersive -= 1

		bridges = nb_bridges_me - nb_bridges_opponent
		pawns = nb_pawn_blocked_opponent - nb_pawn_blocked_me
		safe_pawns = nb_pawn_safe_me - nb_pawn_safe_opponent
		near_center = nb_near_center_opponent - nb_near_center_me
		available_moves = nb_available_moves_me - nb_available_moves_opponent

		return bridges*2 + pawns*1e3 + safe_pawns + dispersive*1e-3 + near_center * 1e-2 + available_moves