import yaml
import copy
import gym
from gym import spaces
import numpy as np
import sys
import os
from scipy.spatial.distance import euclidean
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from physical_env.network.NetworkIO import NetworkIO
from physical_env.mc.MobileCharger import MobileCharger
from scipy.integrate import quad, dblquad
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from controller.Q_learning import Q_learning,init_qtable,init_list_action
# def func(x, hX):
#     km = x * 2 / (-2 * hX * 2)
#     return np.exp(km)

    
class WRSN(gym.Env):
    def __init__(self, scenario_path, agent_type_path, num_agent):
        self.scenario_io = NetworkIO(scenario_path)
        with open(agent_type_path, "r") as file:
            self.agent_phy_para = yaml.safe_load(file)
        self.num_agent = num_agent
        self.agents_process = [None for _ in range(num_agent)]

        
    def reset(self):
        self.env, self.net = self.scenario_io.makeNetwork()
        self.net_process = self.env.process(self.net.operate())
        self.agents = [MobileCharger(copy.deepcopy(self.net.baseStation.location), self.agent_phy_para) for _ in range(self.num_agent)]
        for id, agent in enumerate(self.agents):
            agent.env = self.env
            agent.net = self.net
            agent.id = id
            agent.cur_phy_action = [self.net.baseStation.location[0], self.net.baseStation.location[1], 0, False]
            agent.q_learning.reset()
            self.agents_process[id] = self.env.process(self.agents[id].operate_step(copy.deepcopy(agent.cur_phy_action)))

        return
    
    def choose_request(self):
        if len(self.net.wait_request) > 0:
            self.net.wait_request.pop()
            action = []
            Q_value = []
            for agent in self.agents:
                tmp1, tmp2 = agent.q_learning.update()
                action.append(tmp1)
                Q_value.append(tmp2)

            max_value = np.max(Q_value)
            max_id = np.random.choice([id for id, value in enumerate(Q_value) if value == max_value])

            self.agents[max_id].q_learning.choose_next_state()
            self.agents[max_id].charge_queue.append(action[max_id])
        
    
    def random_agent_action(self):
        if len(self.net.wait_request) > 0:
            id = np.random.randint(0, len(self.agents))
            
            node_id = np.random.randint(0, 82)
            action = [self.net.listNodes[node_id].location[0], self.net.listNodes[node_id].location[1], 20, True]
            self.net.wait_request.pop()
            self.agents[id].charge_queue.append(action)

    def step(self):
        for id, agent in enumerate(self.agents):
            if len(agent.charge_queue) > 0 and agent.cur_phy_action[2] == 0:
                action = agent.charge_queue[0]
                agent.charge_queue.pop(0)
                print (id, action)
                self.agents_process[id] = self.env.process(agent.operate_step(action))
            else:
                agent.location
                self.agents_process[id] = self.env.process(agent.operate_step([agent.location[0], agent.location[1], 0, False]))

        general_process = self.net_process
        for id, agent in enumerate(self.agents):
            if agent.status != 0:
                general_process = general_process | self.agents_process[id]

        self.env.run(until=general_process)
        
        
    def run(self):
        self.env.run(until=self.net_process)