import numpy as np
import math
from scipy.spatial.distance import euclidean

def init_qtable(n_actions):
    return np.zeros((n_actions, n_actions), dtype=float)

def init_list_action(n_actions, net):
    list_action = []
    list_action.append([net.baseStation.location[0], net.baseStation.location[1]])
    for i in range(0, n_actions-1):
        list_action.append([net.listNodes[i].location[0], net.listNodes[i].location[1]])
    return np.asarray(list_action)
    

class Q_learning:
    def __init__(self, mc, epsilon=0.2, alpha=0.5, gamma=0.6, theta = 0.1):
        self.mc = mc
        self.reward_max = 0.0
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.theta = theta
    
    def reset(self):
        self.n_actions = len(self.mc.net.listNodes) + 1
        self.action_list = init_list_action(self.n_actions, self.mc.net)
        #self.q_table = init_qtable(self.n_actions)
        self.charging_time = [0.0 for _ in self.action_list]
        self.reward = [0.0 for _ in self.action_list]
        self.state = 0

    def update(self, charging_time):
        self.caculate_reward(charging_time)
        
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
        self.choose_next_state()

        action = [self.action_list[max_state][0], self.action_list[max_state][1], charging_time, True]
        destination = [self.action_list[max_state][0], self.action_list[max_state][1]]
        return destination, q_max_value, self.state

    def choose_next_state(self):
        #print(self.q_table[self.state])
        self.state = np.argmax(self.q_table[self.state])


    def q_max(self):
        q_next_state = [max(row) for index, row in enumerate(self.q_table)]
        return np.asarray(q_next_state)
    
    def caculate_reward(self, charging_time):
        energy_factor = np.array([0.0 for _ in self.action_list], dtype=float)
        priority_factor = np.array([0.0 for _ in self.action_list], dtype=float)
        target_monitoring_factor = np.array([0.0 for _ in self.action_list], dtype=float)
        for id, _ in enumerate(self.q_table):
            reward = self.reward_function(id, charging_time[id])
            energy_factor[id] = reward[0]
            priority_factor[id] = reward[1]
            target_monitoring_factor[id] = reward[2]

        energy_factor = energy_factor / np.sum(energy_factor)
        priority_factor = priority_factor / np.sum(priority_factor)
        target_monitoring_factor = target_monitoring_factor / np.sum(target_monitoring_factor)

        self.reward = energy_factor + priority_factor + target_monitoring_factor

    def reward_function(self, state,charging_time):
        # connected_nodes = np.array([node for node in self.mc.net.listNodes 
        #                             if euclidean(node.location, self.action_list[state]) <= self.mc.chargingRange])
        p = np.array([self.mc.alpha / (euclidean(self.action_list[state], self.mc.net.listNodes[node_id].location)+ self.mc.beta)**2
                         for node_id in self.mc.net.list_request])
        e = np.array([self.mc.net.listNodes[node_id].energyCS for node_id in self.mc.net.list_request])
        E = np.array([self.mc.net.listNodes[node_id].energy for node_id in self.mc.net.list_request])
        w, nb_target_alive = self.prio_weight(state, charging_time)
        targets = []


        t = nb_target_alive/len(self.mc.net.listTargets)
        # energy factor
        energy_factor = np.sum(e*p/E)
        # priority factor
        priority_factor = np.sum(w*p)
        # target monitoring factor
        target_monitoring_factor = t

        #print (self.action_list[state], energy_factor, priority_factor, target_monitoring_factor)
        
        return energy_factor, priority_factor, target_monitoring_factor

    def get_path(self, node_id):
        path = [node_id]
        if euclidean(self.mc.net.listNodes[node_id].location, self.mc.net.baseStation.location) <= self.mc.net.listNodes[node_id].com_range:
            path.append(-1)
        else:
            receiver = self.mc.net.listNodes[node_id].find_receiver()
            if receiver != None:
                path.extend(self.get_path(receiver.id))
        return path

    def get_all_path(self):
        list_path = []
        for target in self.mc.net.listTargets:
            for node in self.mc.net.listNodes:
                if target in  node.listTargets:
                    list_path.append(self.get_path(node.id))
        return list_path
    
    def prio_weight(self,state,charging_time):
        p = np.array([self.mc.alpha / (euclidean(self.action_list[state], self.mc.net.listNodes[node_id].location) + self.mc.beta) ** 2
                      for node_id in self.mc.net.list_request])
        all_path=self.get_all_path()
        time_move = euclidean(self.action_list[self.state],self.action_list[state])/self.mc.velocity
        list_dead =[]
        w = [0 for _ in self.mc.net.list_request]
        for request_id, node_id in enumerate(self.mc.net.list_request):
            temp = (self.mc.net.listNodes[node_id].energy - time_move * self.mc.net.listNodes[node_id].energyCS
                    + charging_time *(p[request_id]-self.mc.net.listNodes[node_id].energyCS))
            if temp < self.mc.net.listNodes[node_id].threshold :
                list_dead.append(node_id)
        for request_id , node_id in enumerate(self.mc.net.list_request):
            nb_path=0
            for path in all_path:
                if node_id in path :
                    nb_path +=1
            w[request_id]=nb_path

        total_weight = sum(w)
        w = np.asarray([i/total_weight for i in w])
        nb_target_alive =0
        for path in all_path:
            if -1 in path and not (set(list_dead) & set(path)):
                nb_target_alive +=1
        return w, nb_target_alive


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

