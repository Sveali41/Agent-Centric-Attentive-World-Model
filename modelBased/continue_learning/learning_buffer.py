import random
import numpy as np

class EnvLearningBuffer:
    def __init__(self, max_size=100):
        self.list = []
        self.max_size = max_size

    def add(self, entity: dict):
        """Add one environment entry (dict)."""
        self.list.append(entity)
        if len(self.list) > self.max_size:
            self.list = self.list[-self.max_size:]  # Keep only the newest `max_size` entries.

    def remove(self, env_map):
        """
        Remove the entry whose `map` matches the provided `env_map`.
        """
        for i, entry in enumerate(self.list):
            if np.array_equal(entry.get('map'), env_map):
                del self.list[i]
                print(f"Removed environment from buffer at index {i}")
                return
        print("Environment not found in buffer.")

    def get_all(self):
        return self.list
    
    def sample(self):
        if len(self.list) == 0:
            return None
        return random.choice(self.list)

    def __len__(self):
        return len(self.list)
