import numpy as np

M = 1.0
m = 0.1
l = 0.5
g = 9.81
dt = 0.02

# Continuous linear system about upright (theta=0, x=0):
# State x = [p, v, th, w]^T
# p_dot = v
# v_dot = u/M - m*g*th/M  (approx)
# th_dot = w
# w_dot = g/l * th - u/(M*l) (approx)
# Let's derive exact Jacobian:
# theta_acc = (g*sin(th) - cos(th)*(u + m*l*w^2*sin(th))/(M+m)) / (l*(4/3 - m*cos^2(th)/(M+m)))
# At th=0, w=0, u=0:
# denom = l * (4/3 - m/(M+m)) = 0.5 * (4/3 - 0.1/1.1) = 0.5 * (1.3333 - 0.0909) = 0.5 * 1.2424 = 0.6212
# d(theta_acc)/d(th) = g / denom = 9.81 / 0.6212 = 15.792
# d(theta_acc)/du = -1 / ((M+m) * denom) = -1 / (1.1 * 0.6212) = -1.4634
# pos_acc = (u - m*l*theta_acc) / (M+m)
# d(pos_acc)/d(th) = - (m*l / (M+m)) * 15.792 = - (0.05 / 1.1) * 15.792 = -0.7178
# d(pos_acc)/du = (1 - m*l * (-1.4634)) / (M+m) = (1 + 0.05*1.4634) / 1.1 = 1.073 / 1.1 = 0.9756

A = np.array([
    [0, 1, 0, 0],
    [0, 0, -0.7178, 0],
    [0, 0, 0, 1],
    [0, 0, 15.792, 0]
])
B = np.array([
    [0],
    [0.9756],
    [0],
    [-1.4634]
])

print("A matrix:\n", A)
print("B matrix:\n", B)

# Solve continuous LQR
import scipy.linalg
Q = np.diag([5.0, 1.0, 40.0, 2.0])
R = np.array([[0.1]])

P = scipy.linalg.solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P
print("Optimal LQR Gains K:", K)

# Check eigenvalues of A - B @ K
eigs = np.linalg.eigvals(A - B @ K)
print("Closed loop eigenvalues:", eigs)
