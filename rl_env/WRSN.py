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

    
class WRSN(gym.Env):
    def __init__(self, scenario_path, agent_type_path, num_agent):
        self.scenario_io = NetworkIO(scenario_path)
        with open(agent_type_path, "r") as file:
            self.agent_phy_para = yaml.safe_load(file)
        self.num_agent = num_agent
        self.agents_input_action = [None for _ in range(num_agent)]
        self.agents_process = [None for _ in range(num_agent)]
        self.agents_action = [None for _ in range(num_agent)]
        self.reset()
    
    def reset(self):
        self.env, self.net = self.scenario_io.makeNetwork()
        self.net_process = self.env.process(self.net.operate()) & self.env.process(self.update_reward())
        self.agents = [MobileCharger(copy.deepcopy(self.net.baseStation.location), self.agent_phy_para) for _ in range(self.num_agent)]
        for id, agent in enumerate(self.agents):
            agent.env = self.env
            agent.net = self.net
            agent.id = id

        if self.net.alive == 1:
            tmp_terminal = False
        else:
            tmp_terminal = True

        new_request = self.net.list_request[0]
        self.net.list_request.pop(0)
        


            

    def reset(self):
        self.env, self.net = self.scenario_io.makeNetwork()
        self.net_process = self.env.process(self.net.operate()) & self.env.process(self.update_reward())
        self.agents = [MobileCharger(copy.deepcopy(self.net.baseStation.location), self.agent_phy_para) for _ in range(self.num_agent)]
        for id, agent in enumerate(self.agents):
            agent.env = self.env
            agent.net = self.net
            agent.id = id
          
        if self.net.alive == 1:
            tmp_terminal = False
        else:
            tmp_terminal = True

        for id, agent in enumerate(self.agents):
            if euclidean(agent.location, agent.cur_phy_action[0:2]) < self.epsilon and agent.cur_phy_action[2] == 0:
                return {"agent_id":id, 
                        "prev_state": self.agents_prev_state[id],
                        "input_action": self.agents_input_action[id],
                        "action":self.agents_action[id], 
                        "reward": 0.0,
                        "state": self.agents_prev_state[id],
                        "terminal":tmp_terminal,
                        "info": [self.net, self.agents]}
        return {"agent_id":None, 
                "prev_state": None,
                "input_action": None,
                "action": None,
                "reward": None,
                "state": None,
                "terminal":tmp_terminal,
                "info": [self.net, self.agents]}
    
    
    def step(self, agent_id, input_action):
        if agent_id is not None:
            action = np.array(input_action)
            self.agents_input_action[agent_id] = action.copy()
            if self.density_map:
                if not (np.all((action >= 0) & (action <= 1)) and np.isclose(np.sum(action), 1)):
                    action = np.exp(action)
                    action = action / (np.sum(action) + self.epsilon)
                action = self.density_map_to_action(action, agent_id)

            action = np.clip(action, self.action_space.low, self.action_space.high)

            self.agents_action[agent_id] = action
            self.agents_process[agent_id] = self.env.process(self.agents[agent_id].operate_step(self.translate(agent_id, action)))
            self.agents_prev_state[agent_id] = self.get_state(agent_id)
            self.agents_prev_fitness[agent_id] = self.get_network_fitness()

        general_process = self.net_process
        for id, agent in enumerate(self.agents):
            if agent.status != 0:
                general_process = general_process | self.agents_process[id]
        self.env.run(until=general_process)
        if self.net.alive == 0:
            return {"agent_id":None, 
                    "prev_state": None,
                    "input_action": None,
                    "action":None, 
                    "reward": None,
                    "state": None,
                    "terminal":True,
                    "info": [self.net, self.agents]}
        for id, agent in enumerate(self.agents):
            if euclidean(agent.location, agent.cur_phy_action[0:2]) < self.epsilon and agent.cur_phy_action[2] == 0:
                return {"agent_id": id, 
                        "prev_state": self.agents_prev_state[id],
                        "input_action":self.agents_input_action[id], 
                        "action":self.agents_action[id], 
                        "reward": self.get_reward(id),
                        "state": self.get_state(id), 
                        "terminal": False,
                        "info": [self.net, self.agents]}
