import matplotlib.pyplot as plt
import numpy as np
import pickle

with open('q_table.pkl', 'rb') as f:
    q = pickle.load(f)


for i in range(3):
    plt.imshow(q[i], cmap='viridis', interpolation='nearest')
    if i == 0:
        plt.colorbar()  
    plt.title('agent 1')  
    plt.savefig(f'agent{i+1}')
