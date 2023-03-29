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
		if state.game_over():
			return True
		if depth == 1:
			return True
		return False

	"""
	The evaluate function must return an integer value
	representing the utility function of the board.
	"""
	def evaluate(self, state: pontu_state.PontuState):
		# Sum of the number of missing bridges adjacent to each of the opponent’s pawns
		nb_bridges = 0
		for i in range(0, state.size - 2):
			for bridge in state.adj_bridges(1 - self.id, i).values():
				if bridge:
					nb_bridges += 1
		# print("Evaluate : ", 12 - nb_bridges)    
		return (state.size - 2) * 4 - nb_bridges
     