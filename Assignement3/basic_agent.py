
# TODO: renformat the imports for inginious
from pontu_game.agent import AlphaBetaAgent
import pontu_game.minimax as minimax


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
  def successors(self, state):
    # Seems to be to easy... Have I missed something?
    actions = state.get_current_player_actions()
    for action in actions:
        yield (action, state.copy().result(action))

  """
  The cutoff function returns true if the alpha-beta/minimax
  search has to stop and false otherwise.
  """
  def cutoff(self, state, depth):
    pass

  """
  The evaluate function must return an integer value
  representing the utility function of the board.
  """
  def evaluate(self, state):
    pass
