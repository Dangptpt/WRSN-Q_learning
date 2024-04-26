import copy
import numpy as np
class Network:
    def __init__(self, env, listNodes, baseStation, listTargets, max_time):
        self.env = env
        self.listNodes = listNodes
        self.baseStation = baseStation
        self.listTargets = listTargets
        self.targets_active = [1 for _ in range(len(self.listTargets))]
        self.alive = 1
        self.list_request = []
        self.wait_request = []
        self.dead_node = []
        # Setting BS and Node environment and network
        baseStation.env = self.env
        baseStation.net = self
        self.max_time = max_time

        self.frame = np.array([self.baseStation.location[0], self.baseStation.location[0], self.baseStation.location[1], self.baseStation.location[1]], np.float64)
        it = 0
        for node in self.listNodes:
            node.env = self.env
            node.net = self
            node.id = it
            it += 1
            self.frame[0] = min(self.frame[0], node.location[0])
            self.frame[1] = max(self.frame[1], node.location[0])
            self.frame[2] = min(self.frame[2], node.location[1])
            self.frame[3] = max(self.frame[3], node.location[1])
        self.nodes_density = len(self.listNodes) / ((self.frame[1] - self.frame[0]) * (self.frame[3] - self.frame[2]))
        it = 0

        # Setting name for each target
        for target in listTargets:
            target.id = it
            it += 1
         
    # Function is for setting nodes' level and setting all targets as covered
    def setLevels(self):
        for node in self.listNodes:
            node.level = -1
        tmp1 = []
        tmp2 = []
        for node in self.baseStation.direct_nodes:
            if node.status == 1:
                node.level = 1
                tmp1.append(node)

        for i in range(len(self.targets_active)):
            self.targets_active[i] = 0

        while True:
            if len(tmp1) == 0:
                break
            # For each node, we set value of target covered by this node as 1
            # For each node, if we have not yet reached its neighbor, then level of neighbors equal this node + 1
            for node in tmp1:
                for target in node.listTargets:
                    self.targets_active[target.id] = 1
                for neighbor in node.neighbors:
                    if neighbor.status == 1 and neighbor.level == -1:
                        tmp2.append(neighbor)
                        neighbor.level = node.level + 1

            # Once all nodes at current level have been expanded, move to the new list of next level
            tmp1 = tmp2[:]
            tmp2.clear()
        return

    def trigger(self):
        for node in self.net.listNodes:
            if node.status == 0 and self.dead_node[node.id] == 0:
                self.dead_node[node.id] = 1
                return True
        return False
            
    def operate(self, t=1):

        for node in self.listNodes:
            self.env.process(node.operate(t=t))
        self.env.process(self.baseStation.operate(t=t))
        while True:
            yield self.env.timeout(t / 10.0)
            # if self.trigger() == True:
            #     self.setLevels()
            self.setLevels()
            self.alive = self.check_targets()
            yield self.env.timeout(9.0 * t / 10.0)
            
            for node in self.listNodes:
                if node.is_request == True:
                    check = False
                    for request in self.list_request:
                        if request == node.id:
                            check = True
                            break
                    if check == False:
                        self.list_request.append(node.id)
                        self.wait_request.append(node.id)

            removes = []
            for node in self.list_request:
                if self.listNodes[node].is_request == False:
                    removes.append(node)
            for node in removes:
                self.list_request.remove(node)
            
            # if len(self.list_request) > 0 and len(self.wait_request) == 0:
            #     self.wait_request = self.list_request.copy()
            if self.alive == 0 or self.env.now >= self.max_time:
                break   
            
            # if len(self.wait_request) > 0 and len(self.list_request) > 0:
            #     break
        return

    # If any target dies, value is set to 0
    def check_targets(self):
        return min(self.targets_active)
    
    def check_nodes(self):
        tmp = 0
        for node in self.listNodes:
            if node.status == 0:
                tmp += 1
        return tmp
