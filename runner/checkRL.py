import sys
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from rl_env.WRSN import WRSN
import matplotlib.pyplot as plt
import numpy as np
import pickle

def log(net):
    # If you want to print something, just put it here. Do not fix the core code.
    while True:
        if net.env.now % 100 == 0:
            print (net.env.now)
        yield net.env.timeout(1.0)

network = WRSN(scenario_path="physical_env/network/network_scenarios/hanoi1000n50.yaml"
               ,agent_type_path="physical_env/mc/mc_types/default.yaml"
               ,num_agent=3)


n_episode = 2

with open('q_table.pkl', 'rb') as f:
    q = pickle.load(f)

# q = []
# for i in range(3):
#     q.append(np.zeros((83, 83), dtype=float))
# with open('q_table83.pkl', 'wb') as f:
#     pickle.dump(q, f)

# for eps in range(n_episode):
#     network.reset()
#     network.env.process(log(network.net)) 
#     for id, agent, in enumerate(network.agents):
#         agent.q_learning.q_table = q[id]
#     while True:
#         if network.net.alive == 1:
#             network.step()
#         else:
#             for i, node in enumerate(network.net.listNodes):
#                 print (i, node.location, node.energy)
#             for id, agent, in enumerate(network.agents):
#                 q[id] = agent.q_learning.q_table 
#             break

#     with open('q_table.pkl', 'wb') as f:
#         pickle.dump(q, f)
#     print(network.net.env.now)





for i in range(3):
    plt.imshow(q[i], cmap='viridis', interpolation='nearest')
    if i == 0:
        plt.colorbar()  
    plt.title('agent 1')  
    plt.savefig(f'agent{i+1}')
