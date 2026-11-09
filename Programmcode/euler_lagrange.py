from sympy import *
from sympy.core.function import AppliedUndef
init_printing()

t = symbols('t')
G, m = symbols('G M', positive=True)

# Koordinaten des Teilchens
r     = Function('R')(t)
theta = Function('THETA')(t)
phi   = Function('PHI')(t)

e_x, e_y, e_z = symbols('e_x e_y e_z')

m_i = IndexedBase('m')
r_i = IndexedBase('r')
theta_i = IndexedBase('theta')
phi_i = IndexedBase('phi')

# Abstand |r - r_i|
def abstand(i):
    return sqrt(r**2 + r_i[i]**2 - 2*r*r_i[i]*(
        sin(theta)*sin(theta_i[i])*cos(phi - phi_i[i]) +
        cos(theta)*cos(theta_i[i])
    ))

def potentielle_Energie(i):
    return (-G * m * m_i[i]/abstand(i))

# Lagrangefunktion L = T - V
T = Rational(1,2)*m*(r.diff(t)**2 
    + r**2 * theta.diff(t)**2 
    + r**2 * sin(theta)**2 * phi.diff(t)**2)

i = symbols("i", integer=True)
N = symbols("N", integer=True)

V = Sum(-G * m * m_i[i]/ sqrt(r**2 + r_i[i]**2 - 2*r*r_i[i]*(
        sin(theta)*sin(theta_i[i])*cos(phi - phi_i[i]) +
        cos(theta)*cos(theta_i[i])
    )) , (i, 1, N))

L = T - V

# Euler-Lagrange 
def euler_lagrange(L, q):
    return diff(diff(L, q.diff(t)), t) - diff(L, q)


EL_r = euler_lagrange(L, r)

bewegungsgleichung_r = solve(EL_r, r.diff(t, 2))[0]
print(latex(bewegungsgleichung_r))
bewegungsgleichung_r


EL_theta = euler_lagrange(L, theta)

bewegungsgleichung_theta = solve(EL_theta, theta.diff(t, 2))[0]
print(latex(bewegungsgleichung_theta))
bewegungsgleichung_theta


EL_phi = euler_lagrange(L, phi)

bewegungsgleichung_phi = solve(EL_phi, phi.diff(t, 2))[0]
print(latex(bewegungsgleichung_phi))
bewegungsgleichung_phi