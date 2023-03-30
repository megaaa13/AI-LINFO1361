from agent import Agent
from neuronal import *
import torch

class MyAgent(Agent):
  
  def __init__(self):
    # Initialize an empty model
    self.model = NeuralNet(52, 100, 17)
    
    # Load the model from the file
    self.model.load_state_dict(torch.load('best_model.pt'))
    self.model.eval()

  def get_action(self, state, last_action, time_left):
    input_state = NeuralHelpers.state_to_tensor(state)
    input_state = torch.from_numpy(input_state).float()
    output = self.model()
    return output
  
  def get_name(self):
    return 'neuronal agent'