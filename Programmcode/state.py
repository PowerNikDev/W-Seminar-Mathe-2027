import numpy as np
from helpers import to_spherical, cartesian_velocity_to_spherical

class State:
    def __init__(self, location_vectors, velocities, masses, G):
        self.location_vectors = location_vectors
        self.velocities = velocities
        self.masses = masses
        self.G = G
        self.N = len(location_vectors)

    def get_state_vector(self):
        return (self.location_vectors, self.velocities)

    def set_state_vector(self, state_vector):
        self.location_vectors = state_vector[0]
        self.velocities = state_vector[1]

def FIGURE_EIGHT():
    offset = np.array([5.0, 0.0, 0.0])  #  5 Einheiten in x-Richtung

    b1 = np.array([0.9700436, -0.24308753, 0.0]) + offset
    b2 = np.array([-0.9700436,  0.24308753, 0.0]) + offset
    b3 = np.array([0.0, 0.0, 0.0]) + offset  # exakt im "lokalen" Ursprung

    b1 = to_spherical(b1)
    b2 = to_spherical(b2)
    b3 = to_spherical(b3)

    location_vectors = np.array([b1, b2, b3])

    v1 = np.array([0.466203685,  0.43236573, 0.0])
    v2 = np.array([0.466203685, 0.43236573, 0.0])
    v3 = np.array([-0.93240737, -0.86473146, 0.0])

    velocities = np.array([
        cartesian_velocity_to_spherical(v1, b1),
        cartesian_velocity_to_spherical(v2, b2),
        cartesian_velocity_to_spherical(v3, b3)
    ])

    masses = np.array([1.0, 1.0, 1.0])
    return State(location_vectors, velocities, masses, 1)


def TWO_MOONS():
    offset = np.array([10.0, 0.0, 0.0])
    b1 = np.array([0.0, 0.0, 0.0]) + offset
    b2 = np.array([5.0, 0.0, 0.0]) + offset
    b3 = np.array([-5.0, 0.0, 0.0]) + offset
    print (b3 + offset)
    b1 = to_spherical(b1)
    b2 = to_spherical(b2)
    b3 = to_spherical(b3)

    location_vectors = np.array([b1, b2, b3])

    v1 = np.array([0.0, 0.0, 0.0])
    v2 = np.array([0.0, 5.0, 0.0])
    v3 = -v2.copy()

    velocities = np.array([
            cartesian_velocity_to_spherical(v1, b1),
            cartesian_velocity_to_spherical(v2, b2),
            cartesian_velocity_to_spherical(v3, b3)
        ])

    masses = np.array([80.0, 1.0, 1.0])
    return State(location_vectors, velocities, masses, 1)


def RANDOM_BODIES(n=20, radius=10.0, max_velocity=1.0, seed=None):
    if seed is not None:
        np.random.seed(seed)

    offset = np.array([radius * 0, 0.0, 0.0])  # Ursprung vermeiden

    phi_rand   = np.random.uniform(0, 2 * np.pi, n)
    cos_theta  = np.random.uniform(-1, 1, n)
    sin_theta  = np.sqrt(1 - cos_theta**2)
    r_rand     = radius * np.cbrt(np.random.uniform(0, 1, n))  # gleichmäßig im Volumen

    positions_cart = np.column_stack([
        r_rand * sin_theta * np.cos(phi_rand),
        r_rand * sin_theta * np.sin(phi_rand),
        r_rand * cos_theta
    ]) + offset

    # Zufällige Geschwindigkeiten mit Betrag <= max_velocity
    vel_phi   = np.random.uniform(0, 2 * np.pi, n)
    vel_cos_t = np.random.uniform(-1, 1, n)
    vel_sin_t = np.sqrt(1 - vel_cos_t**2)
    vel_mag   = np.random.uniform(0, max_velocity, n)

    velocities_cart = np.column_stack([
        vel_mag * vel_sin_t * np.cos(vel_phi),
        vel_mag * vel_sin_t * np.sin(vel_phi),
        vel_mag * vel_cos_t
    ])

    # Umrechnung in Kugelkoordinaten
    location_vectors = np.array([to_spherical(p) for p in positions_cart])
    velocities       = np.array([
        cartesian_velocity_to_spherical(velocities_cart[i], location_vectors[i])
        for i in range(n)
    ])

    masses = np.ones(n)
    return State(location_vectors, velocities, masses, G=1)