import random
import numpy as np

class EnvLearningBuffer:
    def __init__(self, max_size=100):
        self.list = []
        self.max_size = max_size

    def add(self, entity: dict):
        """添加一个环境 entry（dict）"""
        self.list.append(entity)
        if len(self.list) > self.max_size:
            self.list = self.list[-self.max_size:]  # 保留最新的 max_size 个

    def remove(self, env_map):
        """
        从 buffer 中移除与给定 env_map 相同的 entry
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
