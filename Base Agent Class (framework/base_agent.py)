from abc import ABC, abstractmethod
import numpy as np

class BaseAgent(ABC):
    """Base class for all AI agents; users must implement core methods"""
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

    @abstractmethod
    def act(self, state):
        """Select an action based on the state"""
        pass

    @abstractmethod
    def learn(self, state, action, reward, next_state, done):
        """Update the agent's strategy"""
        pass

    def save(self, path):
        """Save the model (optional implementation)"""
        pass

    def load(self, path):
        """Load the model (optional implementation)"""
        pass
