# TODO: renformat the imports for inginious
from agent import AlphaBetaAgent
import minimax as minimax
import pontu_state as pontu_state


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
		# Seems to be to easy... Have I missed something?
		actions = state.get_current_player_actions()
		for action in actions:
			new_state = state.copy()
			new_state.apply_action(action)
			yield (action, new_state)

	"""
	The cutoff function returns true if the alpha-beta/minimax
	search has to stop and false otherwise.
	"""
	def cutoff(self, state, depth):
		print("Depth: ", depth)
		print("Eval: ", self.evaluate(state) // 1e4)
		if state.game_over():
			return True
		if depth > (self.evaluate(state) // 1e4):
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
			# print(state.adj_bridges(self.id, i).values())
			if sum(1 for v in state.adj_bridges(self.id, i).values() if v == True) >= 2:
				nb_pawn_safe_me += 1

		# Block opponent spawn
		nb_pawn_blocked_opponent = 0
		for i in range(0, state.size - 2):
			if state.is_pawn_blocked(1 - self.id, i):
				nb_pawn_blocked_opponent += 1
		
		# Set a weight to our bridges to make playing offensive more interesting
		# AABBCC.DD
		#   - AA: number of oppenent pawn blocked
		#   - BB: number of my pawn safe
		#   - CC: number of oppenent's bridges
		#   - DD: number of my bridges
		weight = ((state.size - 2) * 4 - nb_bridges_opponent) + nb_bridges_me * 1e-2 \
			+ nb_pawn_safe_me * 1e2 + nb_pawn_blocked_opponent * 1e4
		
		# print("", weight, end="- ")
		return weight
		
