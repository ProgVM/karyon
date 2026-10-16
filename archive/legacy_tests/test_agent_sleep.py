import sys
sys.path.insert(0, '.')
import torch
import karyon_agent

agent = karyon_agent.CoREAgent(device='cpu')
print("Initial manifest:", agent.get_topology_manifest())

# Sleep cycle
res = agent.execute_deep_allostatic_sleep(downscaling_factor=0.01, sprout_probability=1.0)
print("Sleep result:", res)
print("Post-sleep manifest:", agent.get_topology_manifest())
