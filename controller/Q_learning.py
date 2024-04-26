import numpy as np
import math
from scipy.spatial.distance import euclidean

def init_qtable(n_actions):
    return np.zeros((n_actions, n_actions), dtype=float)

def init_list_action(n_actions, net):
    list_action = []
    for i in range(0, n_actions):
        list_action.append([net.listNodes[i].location[0], net.listNodes[i].location[1]])
    return np.asarray(list_action)
    

class Q_learning:
    def __init__(self, mc, epsilon=0.2, alpha=0.1, gamma=0.5, theta = 0.1):
        self.mc = mc
        self.state = 40
        self.reward_max = 0.0
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.theta = theta
    
    def reset(self):
        self.n_actions = len(self.mc.net.listNodes)
        self.action_list = init_list_action(self.n_actions, self.mc.net)
        #self.q_table = init_qtable(self.n_actions)
        self.charging_time = [0.0 for _ in self.action_list]
        self.reward = [0.0 for _ in self.action_list]

    def update(self):
        self.caculate_reward()
        
        # update q row i in q_table
        self.q_table[self.state] = (1 - self.alpha) * self.q_table[self.state] + self.alpha * (self.reward + self.gamma * self.q_max())
        q_max_value = np.max(self.q_table[self.state])
        max_state = np.argmax(self.q_table[self.state])

        # if np.random.uniform(0,1) < self.epsilon :
        #     rand_state = np.random.randint(0, self.n_actions)
        #     rand_q_value = self.q_table[self.state][rand_state]
        #     charging_time = self.charging_time[rand_state]

        #     action = [self.action_list[rand_state][0], self.action_list[rand_state][1], charging_time, True]
        #     return action, rand_q_value, rand_state
        
        charging_time = self.charging_time[max_state]

        action = [self.action_list[max_state][0], self.action_list[max_state][1], charging_time, True]
        return action, q_max_value

    def choose_next_state(self):
        #print(self.q_table[self.state])
        self.state = np.argmax(self.q_table[self.state])


    def q_max(self):
        q_next_state = [max(row) for index, row in enumerate(self.q_table)]
        return np.asarray(q_next_state)
    
    def caculate_reward(self):
        energy_factor = np.array([0.0 for _ in self.action_list], dtype=float)
        priority_factor = np.array([0.0 for _ in self.action_list], dtype=float)
        target_monitoring_factor = np.array([0.0 for _ in self.action_list], dtype=float)
        for id, _ in enumerate(self.q_table):
            reward = self.reward_function(id)
            energy_factor[id] = reward[0]
            priority_factor[id] = reward[1]
            target_monitoring_factor[id] = reward[2]
            self.charging_time[id] = reward[3]  

        energy_factor = energy_factor / np.sum(energy_factor)
        priority_factor = priority_factor / np.sum(priority_factor)
        target_monitoring_factor = target_monitoring_factor / np.sum(target_monitoring_factor)

        self.reward = energy_factor + energy_factor + target_monitoring_factor

    def reward_function(self, state):
        connected_nodes = np.array([node for node in self.mc.net.listNodes 
                                    if euclidean(node.location, self.action_list[state]) <= self.mc.chargingRange])
        p = np.array([self.mc.alpha / (euclidean(self.action_list[state], node.location)+ self.mc.beta)**2
                         for node in connected_nodes])
        e = np.array([node.energyCS for node in connected_nodes])
        E = np.array([node.energy for node in connected_nodes])
        w = np.array([node.num_path + len(node.listTargets) 
                      for node in connected_nodes])
        targets = []
        for target in self.mc.net.listTargets:
            for node in connected_nodes:
                if euclidean(target.location, node.location) <= node.sen_range:
                    targets.append(target)
                    break

        t = len(targets) / len(self.mc.net.listTargets)
        # energy factor
        energy_factor = np.sum(e*p/E)
        # priority factor
        priority_factor = np.sum(w*p)
        # target monitoring factor
        target_monitoring_factor = t
        
        charging_time = self.get_charging_time(state=state)
        #print (self.action_list[state], energy_factor, priority_factor, target_monitoring_factor)
        
        return energy_factor, priority_factor, target_monitoring_factor, charging_time 


    def get_charging_time(self, state):

        energy_critical = self.mc.net.listNodes[0].eth + self.theta * self.mc.net.listNodes[0].capacity
        
        positive_critical_nodes = []  # list of node which critical and positive charge
        negative_normal_nodes = []  # list of node which normal and negative charge
        
        for node in self.mc.net.listNodes:
            if node.energy < energy_critical and node.energyCS - node.energyRR < 0:
                positive_critical_nodes.append(node)
            if node.energy > energy_critical and node.energyCS - node.energyRR > 0:
                negative_normal_nodes.append(node)
        
        ta = []

        for node in positive_critical_nodes:
            t_move = euclidean(self.mc.cur_phy_action[0:2], self.action_list[state]) / self.mc.velocity
            e = node.energy
            e = max (node.threshold, e - t_move*node.energyCS + t_move*node.energyRR)
            charging_rate = self.mc.alpha / (euclidean(self.mc.cur_phy_action[0:2], node.location)+ self.mc.beta)
            ta.append(float(max(energy_critical - e, 0) / charging_rate))

        tb = []
        for node in negative_normal_nodes:
            t_move = euclidean(self.mc.cur_phy_action[0:2], self.action_list[state]) / self.mc.velocity
            e = node.energy
            e = min (node.threshold, e - t_move*node.energyCS + t_move*node.energyRR)
            charging_rate = self.mc.alpha / (euclidean(self.mc.cur_phy_action[0:2], node.location)+ self.mc.beta)
            tb.append(float(max(energy_critical - e, 0) / charging_rate))
        
        ta.sort()
        tb.sort()
        ta.append(999999)
        tb.append(999999)
        t_optimal = 0.0
        node_change_state = -9999
        for change1, t_a in enumerate(ta[:-1]):
            for change2, t_b in enumerate(tb):
                if t_a < t_b:
                    if node_change_state < change1 - change2:
                        node_change_state = change1 - change2
                        t_optimal = t_a
                    break
        
        for change2, t_b in enumerate(tb[:-1]):
            for change1, t_a in enumerate(ta):
                if t_b < t_a: 
                    if node_change_state < change1 - change2:
                        node_change_state = change1 - change2
                        t_optimal = t_b
                    break

        return t_optimal

