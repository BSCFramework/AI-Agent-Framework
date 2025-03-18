from abc import ABC, abstractmethod

class BaseEnvironment(ABC):
    """Base class for environments that interact with agents"""
    @abstractmethod
    def reset(self):
        """Reset the environment and return the initial state"""
        pass

    @abstractmethod
    def step(self, action):
        """Take an action and return (next_state, reward, done, info)"""
        pass

    @property
    @abstractmethod
    def state_dim(self):
        """State dimensionality"""
        pass

    @property
    @abstractmethod
    def action_dim(self):
        """Action dimensionality"""
        pass
