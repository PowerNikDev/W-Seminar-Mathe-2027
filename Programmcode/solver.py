from equations_of_motion import all_equations_of_motion_np
import numpy as np
from abc import ABC, abstractmethod
from state import State, FIGURE_EIGHT, TWO_MOONS, RANDOM_BODIES
from recorder import Recorder
from visualize import visualize

class Integrator(ABC):
    def __init__(self):
        self.masses = None
    
    def f(self, state_vector):
        return (state_vector[1].copy(), all_equations_of_motion_np(state_vector, self.masses))


    def add_vectors(self, sv, k, dt):
        return (sv[0] + k[0] * dt, sv[1] + k[1] * dt)

    @abstractmethod
    def integrate(self, state_vector, time_step):
        pass

class RK4Integrator(Integrator):
    def __init__(self):
        super().__init__()

    def integrate(self, state_vector, time_step):
        K1 = self.f(state_vector)
        K2 = self.f(self.add_vectors(state_vector, K1, time_step / 2))
        K3 = self.f(self.add_vectors(state_vector, K2, time_step / 2))
        K4 = self.f(self.add_vectors(state_vector, K3, time_step))

        new_pos = state_vector[0] + time_step/6 * (K1[0] + 2*K2[0] + 2*K3[0] + K4[0])
        new_vel = state_vector[1] + time_step/6 * (K1[1] + 2*K2[1] + 2*K3[1] + K4[1])
        
        return (new_pos, new_vel)


class NBPSolver:
    def __init__(self, state: State, integrator: Integrator):
        self.state = state
        self.integrator = integrator
        self.integrator.masses = state.masses

    def run(self,
            duration: float,
            time_step: float,
            record_interval: float,
            save_path: str = "simulation",
            visualize: bool = True):

        recorder = Recorder(save_path, self.state.masses)
        state_vector = self.state.get_state_vector()
        t = 0.0
        last_record = -record_interval  # erstes Frame sofort aufnehmen

        print(f"Simuliere {duration} Zeiteinheiten mit dt={time_step}...")
        while t < duration:
            state_vector = self.integrator.integrate(state_vector, time_step)
            t += time_step

            if t - last_record >= record_interval:
                recorder.record(t, state_vector)
                last_record = t

        self.state.set_state_vector(state_vector)
        recorder.save()
        print(f"Simulation abgeschlossen. {len(recorder.times)} Frames gespeichert.")

        if visualize:
            self._visualize(recorder)

        return recorder

    def _visualize(self, recorder: Recorder):
        visualize(
            np.array(recorder.times),
            np.array(recorder.positions),
            self.state.N
        )

if __name__ == "__main__":
    solver = NBPSolver(TWO_MOONS(), RK4Integrator())
    solver.run(
        duration = 100,
        time_step=0.0001,
        record_interval=0.01,
        save_path="./Two_Moons",
        visualize=True
    )