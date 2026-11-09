import os
import numpy as np
from state import State

class Recorder:
    def __init__(self, save_path: str, masses):
        self.save_path = self._unique_path(save_path)
        self.times = []
        self.positions = []
        self.velocities = []
        self.masses = masses

    @staticmethod
    def _unique_path(path: str) -> str:
        if not os.path.exists(f"{path}.npz"):
            return path
        counter = 1
        while os.path.exists(f"{path}_{counter}.npz"):
            counter += 1
        return f"{path}_{counter}"

    def record(self, t, state_vector):
        self.times.append(t)
        self.positions.append(state_vector[0].copy())
        self.velocities.append(state_vector[1].copy())

    def save(self):
        np.savez(self.save_path,
         times=np.array(self.times),
         positions=np.array(self.positions),
         velocities=np.array(self.velocities),
         masses=self.masses)  # neu
        print(f"Gespeichert: {self.save_path}.npz")

    @staticmethod
    def load(path):
        data = np.load(path)
        return data['times'], data['positions'], data['velocities'], data['masses']
