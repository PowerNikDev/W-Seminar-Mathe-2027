from math import sin, cos, sqrt, acos, atan2
G = 1
import numpy as np

# [r, theta, phi] --> [x, y, z]
def to_cartesian(sphere_coords):
    r = sphere_coords[0]
    theta = sphere_coords[1]
    phi = sphere_coords[2]

    return [r * sin(theta) * cos(phi), r * sin(theta) * sin(phi), r * cos(theta)]


def total_kinetic_energy(state_vector, masses):
    s = 0
    for i in range(len(state_vector[1])):
        v_x, v_y, v_z = spherical_velocity_to_cartesian(state_vector[1][i], state_vector[0][i])
        s += 0.5 * masses[i] * (v_x**2 + v_y**2 + v_z**2)

    return s


def total_potential_energy(state_vector, masses):
    sum = 0
    n = len(state_vector[0])
    for i in range(n):
            for j in range(i + 1, n):
                dist = distance(state_vector[0][i], state_vector[0][j])
                sum += -1 * G * masses[i] * masses[j] / dist
    return sum


def distance(p1, p2):
    p1 = to_cartesian(p1)
    p2 = to_cartesian(p2)
    diff = np.subtract(p1, p2)
    return sqrt(sum ([x**2 for x in diff]))

def to_spherical(coords):
    r = sqrt(sum(x**2 for x in coords))
    theta = acos(coords[2] / r)
    phi = atan2(coords[1], coords[0])

    return [r, theta, phi]

def cartesian_velocity_to_spherical(v_cartesian, position_spherical):
    r     = position_spherical[0]
    theta = position_spherical[1]
    phi   = position_spherical[2]

    v_x, v_y, v_z = v_cartesian

    # Jacobi-Matrix J (Spalten: d/dr, d/dtheta, d/dphi der Transformation)
    J = np.array([
        [sin(theta)*cos(phi),  r*cos(theta)*cos(phi), -r*sin(theta)*sin(phi)],
        [sin(theta)*sin(phi),  r*cos(theta)*sin(phi),  r*sin(theta)*cos(phi)],
        [cos(theta),          -r*sin(theta),            0                   ]
    ])

    v_spherical = np.linalg.solve(J, np.array([v_x, v_y, v_z]))
    return v_spherical.tolist()  # [dr, dtheta, dphi]

def spherical_velocity_to_cartesian(v_spherical, position_spherical):
    r, theta, phi = position_spherical
    d_r, d_theta, d_phi = v_spherical

    v_x = sin(theta)*cos(phi)*d_r + r*cos(theta)*cos(phi)*d_theta - r*sin(theta)*sin(phi)*d_phi
    v_y = sin(theta)*sin(phi)*d_r + r*cos(theta)*sin(phi)*d_theta + r*sin(theta)*cos(phi)*d_phi
    v_z = cos(theta)*d_r                                           - r*sin(theta)*d_theta
    return [v_x, v_y, v_z]
