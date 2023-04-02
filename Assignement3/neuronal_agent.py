from agent import Agent
from neuronal_torch import *
import torch

class MyAgent(Agent):
  
    def __init__(self):
        # Initialize an empty model
        self.model = NeuralNet(52, 100, 17, 1)
        
        # Load the model from the file
        self.model.load_state_dict(torch.load('model.pt'))
        self.model.eval()

    def get_action(self, state, last_action, time_left):
        return NeuralGenetic.get_action(self.model, state)
  
    def get_name(self):
        return 'neuronal agent'