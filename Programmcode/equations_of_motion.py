from numpy import sin, cos, tan, sqrt
import numpy as np
G = 1


"""Genutzte Version, zusammen mit Claude durch Numpy optimiert"""
def all_equations_of_motion_np(state_vector, masses, G=1.0, eps=1e-10):
    pos = state_vector[0]  # shape (n, 3)
    vel = state_vector[1]  # shape (n, 3)

    r     = pos[:, 0]
    theta = pos[:, 1]
    phi   = pos[:, 2]
    v_r     = vel[:, 0]
    v_theta = vel[:, 1]
    v_phi   = vel[:, 2]
    n = len(r)

    # --- Sichere Nenner ---
    r_safe         = np.where(r < eps, eps, r)
    sin_theta      = np.sin(theta)
    sin_theta_safe = np.where(np.abs(sin_theta) < eps, eps, sin_theta)

    # --- Paarweise Gravitationsbeiträge ---
    ri, rj   = r[:, None],     r[None, :]
    ti, tj   = theta[:, None], theta[None, :]
    pi_, pj  = phi[:, None],   phi[None, :]

    cos_dphi = np.cos(pi_ - pj)
    sin_dphi = np.sin(pi_ - pj)

    # Euklidischer Abstand^2 (n x n)
    dist_sq = (ri**2
               - 2*ri*np.sin(ti)*np.sin(tj)*cos_dphi*rj
               - 2*ri*np.cos(ti)*np.cos(tj)*rj
               + rj**2)

    # d = 0 Warnung
    mask_collision = (np.sqrt(np.maximum(dist_sq, 0)) < eps)
    np.fill_diagonal(mask_collision, False)  # Diagonale ignorieren
    if np.any(mask_collision):
        pairs = np.argwhere(mask_collision)
        for p in pairs:
            if p[0] < p[1]:  # jedes Paar nur einmal
                print(f"Warnung: Kollision zwischen Körper {p[0]} und {p[1]} (d ungefähr 0)")

    # Nenner absichern
    dist_sq_safe = np.where(dist_sq < eps**2, eps**2, dist_sq)
    dist3        = dist_sq_safe ** (3/2)

    # Diagonalmaske (i != j)
    mask = ~np.eye(n, dtype=bool)

    # r-Komponente
    z_r      = -ri + np.sin(ti)*np.sin(tj)*cos_dphi*rj + np.cos(ti)*np.cos(tj)*rj
    grav_r   = G * np.sum(masses[None, :] * z_r   / dist3 * mask, axis=1)

    # theta-Komponente
    z_theta    = rj * (-np.sin(ti)*np.cos(tj) + np.sin(tj)*cos_dphi*np.cos(ti))
    grav_theta = G * np.sum(masses[None, :] * z_theta / dist3 * mask, axis=1)

    # phi-Komponente
    z_phi    = sin_dphi * np.sin(tj) * rj
    grav_phi = G * np.sum(masses[None, :] * z_phi  / dist3 * mask, axis=1)

    # --- Beschleunigungen ---
    a_r = (grav_r
           + r * sin_theta**2 * v_phi**2
           + r * v_theta**2)

    a_theta = (grav_theta / r_safe
               + np.sin(2*theta) * v_phi**2 / 2
               - 2 * v_r * v_theta / r_safe)

    a_phi = (-grav_phi / (r_safe * sin_theta_safe)
             - 2 * v_phi * v_theta * np.cos(theta) / sin_theta_safe
             - 2 * v_phi * v_r / r_safe)

    return np.stack([a_r, a_theta, a_phi], axis=1)



"""Langsamere, von Hand geschriebene Version. Nicht zur Verwendung gedacht. Nur aufgrund der besseren Verständlichkeit inkludiert. Ausgabe und erwartete Eingaben nicht identisch zur oberen Methoden"""
def all_equations_of_motion(zustand, massen):
    n = len(zustand[0]) 
    
    # Positionen und Geschwindigkeiten aus dem Zustandsvektor extrahieren
    r       = [zustand[0][i][0]     for i in range(n)]
    theta   = [zustand[0][i][1]     for i in range(n)]
    phi     = [zustand[0][i][2]     for i in range(n)]
    v_r     = [zustand[1][i][0]     for i in range(n)]
    v_theta = [zustand[1][i][1]     for i in range(n)]
    v_phi   = [zustand[1][i][2]     for i in range(n)]
    m       = massen


    # Gravitationsbeiträge vorberechnen, jedes Paar nur einmal
    grav_r     = [0.0] * n
    grav_theta = [0.0] * n
    grav_phi   = [0.0] * n

    for i in range(n):
        for j in range(i + 1, n):

            distance_squared = (
                r[i]**2
                + r[j]**2
                - 2*r[i]*r[j]*(
                    sin(theta[i])*sin(theta[j])*cos(phi[i]-phi[j])
                    + cos(theta[i])*cos(theta[j])
                )
            )

            distance_squared = max(distance_squared, 0.0)
            distance = sqrt(distance_squared)
            if distance == 0:
                raise ValueError("Zwei Körper befinden sich an derselben Position")
            nenner = distance**3

            # r-Komponente
            grav_r[i] += G * m[j] * (-r[i] + sin(theta[i])*sin(theta[j])*cos(phi[i]-phi[j])*r[j]
                                      + cos(theta[i])*cos(theta[j])*r[j]) / nenner
            grav_r[j] += G * m[i] * (-r[j] + sin(theta[j])*sin(theta[i])*cos(phi[j]-phi[i])*r[i]
                                      + cos(theta[j])*cos(theta[i])*r[i]) / nenner

            # theta-Komponente
            grav_theta[i] += G * m[j] * r[j] * (-sin(theta[i])*cos(theta[j])
                                                  + sin(theta[j])*cos(phi[i]-phi[j])*cos(theta[i])) / nenner
            grav_theta[j] += G * m[i] * r[i] * (-sin(theta[j])*cos(theta[i])
                                                  + sin(theta[i])*cos(phi[j]-phi[i])*cos(theta[j])) / nenner

            # phi-Komponente
            grav_phi[i] += G * m[j] *  sin(phi[i]-phi[j]) * sin(theta[j]) * r[j] / nenner
            grav_phi[j] += G * m[i] *  sin(phi[j]-phi[i]) * sin(theta[i]) * r[i] / nenner

    # Zustandsableitung zusammenbauen
    ableitung = []
    for i in range(n):
        a_r = (grav_r[i]
               + r[i] * sin(theta[i])**2 * v_phi[i]**2
               + r[i] * v_theta[i]**2)

        a_theta = (grav_theta[i] / r[i]
                   + sin(2*theta[i]) * v_phi[i]**2 / 2
                   - 2 * v_r[i] * v_theta[i] / r[i])

        a_phi = (-grav_phi[i] / (r[i] * sin(theta[i]))
                 - 2 * v_phi[i] * v_theta[i] / tan(theta[i])
                 - 2 * v_phi[i] * v_r[i] / r[i])

        # Ableitung der Position = Geschwindigkeit
        ableitung.append([a_r, a_theta, a_phi])

    return ableitung

