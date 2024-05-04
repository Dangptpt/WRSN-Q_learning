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
            
            if np.random.uniform(0,1) < 0.3:
                max_id = np.random.randint(0, len(self.agents))
            else:
                max_value = np.max(Q_value)
                max_id = np.random.choice([id for id, value in enumerate(Q_value) if value == max_value])

            print(self.env.now)
            print (Q_value)
            self.agents[max_id].q_learning.choose_next_state()
            print (self.agents[0].q_learning.state, self.agents[1].q_learning.state, self.agents[2].q_learning.state)
            self.agents[max_id].charge_queue.append(action[max_id])
        
    
    def random_agent_action(self):
        if len(self.net.list_request) > 0:
            id = np.random.randint(0, len(self.agents))
            
            node_id = np.random.randint(0, 82)
            action = [self.net.listNodes[node_id].location[0], self.net.listNodes[node_id].location[1], 20, True]
            self.agents[id].charge_queue.append(action)

    def step(self):            
        n_request = len(self.net.list_request) 

        for id, agent in enumerate(self.agents):
            if n_request > 0 and agent.cur_phy_action[2] == 0:
                charging_time = self.charging_time(id)
                #print(charging_time)
                destination, q_max_value, state = agent.q_learning.update(charging_time)
                if agent.cur_phy_action[3] == True:
                    action = [destination[0], destination[1], charging_time[state], True]
                else:
                    action = [destination[0], destination[1], (3-id)*100, True]
                #action = [destination[0], destination[1], charging_time[state], True]
                print (id, action)
                self.env.process(agent.operate_step(action))

        self.env.run(until=self.env.now + 1)    
        

    def charging_time(self, mc_id):
        time = []
        for i in range(self.agents[mc_id].q_learning.n_actions):
            time.append(self.get_charging_time(self.agents[mc_id], i))
        return time
    
    def get_charging_time(self, mc, state):
        time_move = euclidean(mc.cur_phy_action[0:2], mc.q_learning.action_list[state]) / mc.velocity

        energy_critical = self.net.listNodes[0].eth + 0.1 * self.net.listNodes[0].capacity

        positive_critical_nodes = []  # list of node which critical and positive charge
        negative_normal_nodes = []  # list of node which normal and negative charge

        for node in self.net.listNodes:
            d = euclidean(node.location, mc.q_learning.action_list[state])
            p = (mc.alpha / (d + mc.beta) ** 2)
            p1 = 0
            for other_mc in self.agents:
                d = euclidean(other_mc.cur_phy_action[0:2], node.location)
                if other_mc.chargingRange > d and other_mc.id != mc.id:
                    p1 += (other_mc.alpha / (d + other_mc.beta) ** 2) * other_mc.cur_phy_action[2]
           
            if node.energy - time_move * node.energyCS + p1 < energy_critical and p - node.energyCS > 0:
                positive_critical_nodes.append((node, p, p1))
            if node.energy - time_move * node.energyCS + p1 > energy_critical and p - node.energyCS < 0:
                negative_normal_nodes.append((node, p, p1))
        ta = []

        for node, p, p1 in positive_critical_nodes:
            ta.append((energy_critical - node.energy + time_move * node.energyCS - p1) / (p - node.energyCS))
            
        for node, p, p1 in negative_normal_nodes:
            ta.append((energy_critical - node.energy + time_move * node.energyCS - p1) / (p - node.energyCS))
          
        dead_list = []
        #print(ta)
        for T in ta:
            dead_node = 0
            for node, p, p1 in positive_critical_nodes:
                term = node.energy - time_move * node.energyCS + p1 + (p - node.energyCS) * T
                if term < energy_critical:
                    dead_node += 1
            for node, p, p1 in negative_normal_nodes:
                term = node.energy - time_move * node.energyCS + p1 + (p - node.energyCS) * T
                if term < energy_critical:
                    dead_node += 1
            dead_list.append(dead_node)
        if dead_list:
            arg_min = np.argmin(dead_list)
            return ta[arg_min]
        return 0