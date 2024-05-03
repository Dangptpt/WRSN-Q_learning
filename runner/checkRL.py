import sys
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from rl_env.WRSN import WRSN

def log(net):
    # If you want to print something, just put it here. Do not fix the core code.
    while True:
        if net.env.now % 100 == 0:
            print (net.env.now)
        yield net.env.timeout(1.0)

network = WRSN(scenario_path="physical_env/network/network_scenarios/hanoi1000n50.yaml"
               ,agent_type_path="physical_env/mc/mc_types/default.yaml"
               ,num_agent=3)


n_episode = 1


q = []
for i in range(3):
    q.append(np.zeros((83, 83), dtype=float))
    
for eps in range(n_episode):
    network.reset()
    network.env.process(log(network.net)) 
    for id, agent, in enumerate(network.agents):
        agent.q_learning.q_table = q[id]
    while True:
        if network.net.alive == 1:
            network.step()
        else:
            for i, node in enumerate(network.net.listNodes):
                print (i, node.location, node.energy, node.energyCS, node.energyRR, node.num_path)
            for id, agent, in enumerate(network.agents):
                q[id] = agent.q_learning.q_table 
            break
    print(network.net.env.now)


import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(q[0], q[1], q[2], cmap='viridis') 
plt.savefig("heatmapq1.png")
