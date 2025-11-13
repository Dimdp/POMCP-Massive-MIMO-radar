from pomcp.helper import rand_choice, randint, round
from pomcp.helper import elem_distribution, ucb
from pomcp.belief_tree import BeliefTree
from utils.env_helper import *
import numpy as np
import time

MAX = np.inf

class UtilityFunction():
    @staticmethod
    def ucb1(c):
        def algorithm(action):
            return action.V + c * ucb(action.parent.N, action.N)
        return algorithm
    
    @staticmethod
    def mab_bv1(min_cost, c=1.0):
        def algorithm(action):
            if action.mean_cost == 0.0:
                return MAX
            ucb_value = ucb(action.parent.N, action.N)
            return action.mean_reward / action.mean_cost + c * ((1. + 1. / min_cost) * ucb_value) / (min_cost - ucb_value)
        return algorithm

    @staticmethod
    def sa_ucb(c0):
        def algorithm(action):
            if action.mean_cost == 0.0:
                return MAX
            return action.V + c0 * action.parent.budget * ucb(action.parent.N, action.N)
        return algorithm


class POMCP():
    def __init__(self, model):
        self.model = model
        self.tree = None

        self.simulation_time = None  # in seconds
        self.max_particles = None    # maximum number of particles can be supplied by hand for a belief node
        self.reinvigorated_particles_ratio = None  # ratio of max_particles to mutate 
        self.utility_fn = None

    def add_configs(self, 
                    budget=float('inf'), 
                    initial_belief=None, 
                    simulation_time=0.5,
                    max_particles=500, 
                    reinvigorated_particles_ratio=0.1, 
                    utility_fn='ucb1', 
                    C=0.5):
        
        # acquaire utility function to choose the most desirable action to try
        if utility_fn == 'ucb1':
            self.utility_fn = UtilityFunction.ucb1(C)
        elif utility_fn == 'sa_ucb':
            self.utility_fn = UtilityFunction.sa_ucb(C)
        elif utility_fn == 'mab_bv1':
            if self.model.costs is None:
                raise ValueError('Must specify action costs if utility function is MAB_BV1')
            self.utility_fn = UtilityFunction.mab_bv1(min(self.model.costs), C)

        # other configs
        self.simulation_time = simulation_time
        self.max_particles = max_particles
        self.reinvigorated_particles_ratio = reinvigorated_particles_ratio
        
        # initialise belief search tree
        root_particles = initial_belief.copy()
        self.tree = BeliefTree(budget, root_particles)

    def rollout(self, state, h, depth, max_depth, budget):
        """
        Perform randomized recursive rollout search starting from 'h' util the max depth has been achived
        :param state: starting state's index
        :param h: history sequence
        :param depth: current planning horizon
        :param max_depth: max planning horizon
        :return:
        """
        if depth > 50*max_depth or budget <= 0:
            return 0

        ai = rand_choice(self.model.get_legal_actions(state))
        sj, oj, r, cost = self.model.simulate_action(state, ai)

        return r + self.model.discount * self.rollout(sj, h + [ai, oj], depth + 1, max_depth, budget-cost)
        
    def simulate(self, state, max_depth, depth=0, h=[], parent=None, budget=None):
        """
        Perform MCTS simulation on a POMCP belief search tree
        :param state: starting state's index
        :return:
        """
        # Stop recursion once we are deep enough in our built tree
        if depth > max_depth:
            return 0

        obs_h = None if not h else h[-1]
        node_h = self.tree.find_or_create(h, name=obs_h or 'root', parent=parent,
                                          budget=budget, observation=obs_h)

        # ===== ROLLOUT =====
        # Initialize child nodes and return an approximate reward for this
        # history by rolling out until max depth
        if not node_h.children:
            # always reach this line when node_h was just now created
            for ai in self.model.get_legal_actions(state):
                cost = self.model.cost_function(ai)
                # only adds affordable actions
                if budget - cost >= 0:
                    self.tree.add(h + [ai], name=ai, parent=node_h, action=ai, cost=cost)

            return self.rollout(state, h, depth, max_depth, budget)
        
        legal_actions = self.model.get_legal_actions(state)
        node_actions = node_h.action_map.keys()
        
        for ai in legal_actions:
            if ai not in node_actions:
                cost = self.model.cost_function(ai)
                if budget - cost >= 0:
                    self.tree.add(h + [ai], name=ai, parent=node_h, action=ai, cost=cost)

        # ===== SELECTION =====
        # Find the action that maximises the utility value
        
        np.random.shuffle(node_h.children)
        node_ha = sorted(node_h.children, key=self.utility_fn, reverse=True)[0]
        #node_ha =  sorted([node_h.get_child(ai) for ai in legal_actions], key=self.utility_fn, reverse=True)[0]
        
        # ===== SIMULATION =====
        # Perform monte-carlo simulation of the state under the action
        sj, oj, reward, cost = self.model.simulate_action(state, node_ha.action)
        R = reward + self.model.discount * self.simulate(sj, max_depth, depth + 1, h = h + [node_ha.action, oj],
                                                         parent=node_ha, budget=budget-cost)
        # ===== BACK-PROPAGATION =====
        # Update the belief node for h
        if (depth != 0) and (len(node_h.B) < self.max_particles):
            node_h.B += [state]
        
        node_h.N += 1

        # Update the action node for this action
        node_ha.update_stats(cost, reward)
        node_ha.N += 1
        node_ha.V += (R - node_ha.V) / node_ha.N

        return R

    def solve(self, T, n_simu = 100):
        """
        Solves for up to T steps
        """
        begin = time.time()
        n = 0
        while n < n_simu:
            n += 1
            state = self.tree.root.sample_state()
            self.simulate(state, max_depth=T, h=self.tree.root.h, budget=self.tree.root.budget)
            

    def get_action(self, belief):
        """
        Choose the action maximises V
        'belief' is just a part of the function signature but not actually required here
        """
        root = self.tree.root
        action_vals = [(action.V, action.action) for action in root.children]
        
        return max(action_vals)[1]

    def update_belief(self, belief, action, obs, num_trials = 100):
        """
        Updates the belief tree given the environment feedback.
        extending the history, updating particle sets, etc
        """
        m, root = self.model, self.tree.root
        #####################
        # Find the new root #
        #####################
        new_root = root.get_child(action).get_child(obs)
        print('observations children = ', root.get_child(action).obs_map.keys())
        if new_root is None:
            print(' = = = = = = = = = Warning: node is not in the search tree', [action, obs])
            #log.warning("Warning: {} is not in the search tree".format(root.h + [action, obs]))
            # The step result randomly produced a different observation
            action_node = root.get_child(action)
            if action_node.children:
                # grab any of the beliefs extending from the belief node's action node (i.e, the nearest belief node)
                #log.info('grabing a bearest belief node...')
                new_root = rand_choice(action_node.children)
            else:
                particles = belief.copy()
                new_root = self.tree.add(h=action_node.h + [obs], name=obs, parent=action_node, observation=obs,
                                         particle=particles, budget=root.budget - action_node.cost)
        
        ##################
        # Fill Particles #
        ##################
        
        particle_slots = self.max_particles - len(new_root.B)
        ki = 0
        
        if particle_slots > 0:
            # fill particles by Monte-Carlo using reject sampling
            particles = []
            while len(particles) < particle_slots:
                n_samples = particle_slots - len(particles)
                
                si = root.sample_state(n_samples = n_samples)
                sj, oj, r, cost = self.model.simulate_action_vectorized(si, action)
                obs_to_keep = np.where(obs == oj)[0]
                
                if len(obs_to_keep) > 0 :
                    particles += list(sj[obs_to_keep,:])
                    
                if ki > num_trials:
                    break
                ki += 1
            print('##### POMCP particles with generator model == ', len(particles))
            print('##### size of the belief == ', len(new_root.B))
            new_root.B += particles
            
        
        #####################
        # Advance and Prune #
        #####################
        self.tree.prune(root, exclude=new_root)
        self.tree.root = new_root
        new_belief = self.tree.root.B 

        return new_belief

    
    def draw(self, beliefs):
        """
        Dummy
        """
        pass
