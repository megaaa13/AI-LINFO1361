from agent import Agent
from neuronal_torch import *
import torch

class MyAgent(Agent):
  
    def __init__(self):
        # Initialize an empty model
        self.model = NeuralNet(52, 100, 19, 1)
        
        # Load the model from the file
        state_dict = torch.load('pre_trained.pt', map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def get_action(self, state, last_action, time_left):
        return NeuralGenetic.get_action(self.model, state)
  
    def get_name(self):
        return 'neuronal agent'