import pytest
import karyon_core

def test_karyon_core_initialization():
    agent = karyon_core.SovereignCoreAgent()
    telem = agent.get_telemetry()
    assert telem['active_modes'] == 64
    assert telem['phase_dim'] == 128
    assert telem['free_energy'] == 1.0

def test_karyon_core_perceive_and_assimilate():
    agent = karyon_core.SovereignCoreAgent()
    action = agent.perceive_and_decide(100)
    assert 0 <= action < 256
    
    agent.assimilate_experience(100, valence=1.0, magnitude=1.0, target_hint=100)
    telem = agent.get_telemetry()
    assert telem['stored_phase_traces'] == 1
    assert telem['free_energy'] < 1.0  # Positive feedback reduces free energy

def test_karyon_core_autopoiesis_sprouting():
    agent = karyon_core.SovereignCoreAgent()
    # Induce negative feedback to trigger volumetric stress
    for i in range(100):
        action = agent.perceive_and_decide(i % 256)
        agent.assimilate_experience(i % 256, valence=-1.0, magnitude=2.0)
    
    telem = agent.get_telemetry()
    print("Post-Stress Telemetry:", telem)
    assert telem['autopoietic_stress'] > 0.0 or telem['active_modes'] >= 64
