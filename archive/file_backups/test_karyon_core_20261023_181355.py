"""
Pytest Suite for Karyon Sovereign Core v28.0 (Python / C-ABI Binding)
"""

import math
import pytest
from karyon_core import SovereignKaryonCore

def test_core_instantiation():
    core = SovereignKaryonCore()
    assert core is not None
    assert core.free_energy == 0.0

def test_core_step_and_free_energy():
    core = SovereignKaryonCore()
    torque = core.step(error=0.4, error_rate=0.0, dt=0.02)
    assert torque > 0.0
    assert core.free_energy > 0.0
    expected_fe = 0.5 * (0.4 ** 2)
    assert math.isclose(core.free_energy, expected_fe, rel_tol=1e-4)

def test_core_symmetry():
    c_pos = SovereignKaryonCore()
    c_neg = SovereignKaryonCore()

    t_pos = c_pos.step(0.3, 0.0, 0.02)
    t_neg = c_neg.step(-0.3, 0.0, 0.02)

    assert t_pos > 0.0
    assert t_neg < 0.0
    assert math.isclose(t_pos, -t_neg, rel_tol=1e-3)

def test_core_reset():
    core = SovereignKaryonCore()
    core.step(0.5, 0.2, 0.02)
    assert core.free_energy > 0.0
    core.reset()
    assert core.free_energy == 0.0

def test_core_convergence_closed_loop():
    core = SovereignKaryonCore()
    theta = 0.45
    omega = 0.0
    dt = 0.02
    g = 9.81
    L = 1.0

    for _ in range(5000):
        torque = core.step(theta, omega, dt)
        alpha = (g / L) * math.sin(theta) - torque - 0.1 * omega
        omega += dt * alpha
        theta += dt * omega

    assert abs(theta) < 0.01, f"Theta failed to stabilize: {theta}"
    assert core.free_energy < 1e-4, f"Free energy failed to drop: {core.free_energy}"
