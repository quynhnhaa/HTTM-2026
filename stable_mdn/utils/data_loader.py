from __future__ import print_function
from six.moves import cPickle as pickle
from termcolor import colored
from utils.helper import generate_unique_randoms

import numpy as np
import random


class DataLoader:
    """Trajectory dataset data loader class
    """
    
    def __init__(self, cfg=''):
        """Init and setup
        """
        
        # variables
        self.cfg = cfg
        self.name = cfg.name
        self.train_data_paths = cfg.paths['train_data_paths']
        self.eval_data_paths = cfg.paths['val_data_paths']
        self.test_data_paths = cfg.paths['test_data_paths']
        self.randomize = cfg.train_params['randomize_train_data']
        self.train_data_reduction = cfg.train_params['train_data_reduction']
        self.eval_data_reduction = cfg.train_params['eval_data_reduction']
        self.with_print = cfg.with_print
        
        # data storages
        self.train_data = []
        self.eval_data = []
        self.test_data = []
        self.sample_ids = {'train': [], 'eval': [], 'test': []}
        self.sample_keys = {'train': [], 'eval': [], 'test': []}
        # Tracks the current in-memory ordering because the upstream loader
        # shuffles train arrays in place. Persisting it makes resume equivalent
        # to an uninterrupted run at epoch boundaries.
        self.train_order = None
        
        return
        
        
    def load_train_data(self):
        """Load training data from file
        """
        
        if self.with_print: print(colored(f" Data loader: loading train data: {self.train_data_paths}", 'green'))
        self.train_data = self.load_pkl_data(paths=self.train_data_paths, type='train')
        self.train_order = np.arange(len(self.train_data[0]), dtype=np.int64)
        
        if self.with_print: print(colored(f" Data loader: loaded {len(self.train_data[0])} samples for training", 'green'))
        if self.with_print: print(colored(f" with randomized training data: {self.randomize}", 'green'))
        if self.with_print: print(colored(f" data reduction scale: {self.train_data_reduction}", 'green'))
        
        return
        
        
    def load_eval_data(self):
        """Load train-eval data from file
        """
        
        if self.with_print: print(colored(f" Data loader: loading eval data: {self.eval_data_paths}", 'green'))
        self.eval_data = self.load_pkl_data(paths=self.eval_data_paths, type='eval')
        
        if self.with_print: print(colored(f" Data loader: loaded {len(self.eval_data[0])} samples for train-eval", 'green'))
        if self.with_print: print(colored(f" data reduction scale: {self.eval_data_reduction}", 'green'))
        
        return
        
        
    def load_test_data(self):
        """Load test data from file
        """
        
        if self.with_print: print(colored(f" Data loader: loading test data: {self.test_data_paths}", 'green'))
        self.test_data = self.load_pkl_data(paths=self.test_data_paths, type='test')
        
        if self.with_print: print(colored(f" Data loader: loaded {len(self.test_data[0])} samples for testing", 'green'))
        
        return
    
    
    def get_train_data(self):
        """Get training data for next epoch
        """
        
        # with random shuffle
        if self.randomize:
            
            shuffled_indices = list(range(0, len(self.train_data[0])))
            random.shuffle(shuffled_indices)
            self.train_data[0] = np.take(a=self.train_data[0], indices=shuffled_indices, axis=0)
            self.train_data[1] = np.take(a=self.train_data[1], indices=shuffled_indices, axis=0)
            self.train_order = np.take(a=self.train_order, indices=shuffled_indices, axis=0)
            
        # reduce train data to a random subset
        if self.train_data_reduction < 1.0:
            
            n = int(len(self.train_data[0]) * self.train_data_reduction)
            random_reduced_indices = generate_unique_randoms(count=n, min_value=0, max_value=len(self.train_data[0])-1)
            X = np.take(a=self.train_data[0], indices=random_reduced_indices, axis=0)
            y = np.take(a=self.train_data[1], indices=random_reduced_indices, axis=0)
            
            self.current_train_ids = [self.sample_ids['train'][int(i)] for i in self.train_order[random_reduced_indices]]
            return X, y
        
        # with no modifications
        else:
            
            self.current_train_ids = [self.sample_ids['train'][int(i)] for i in self.train_order]
            return self.train_data[0], self.train_data[1]


    def state_dict(self):
        """Return the mutable loader state required for exact epoch resume."""
        return {
            'schema_version': '1.0',
            'train_order': None if self.train_order is None else self.train_order.copy(),
        }


    def load_state_dict(self, state):
        """Restore the in-place train ordering saved at an epoch boundary."""
        if not state or state.get('train_order') is None:
            return False
        saved_order = np.asarray(state['train_order'], dtype=np.int64)
        if self.train_order is None or len(saved_order) != len(self.train_order):
            raise ValueError(
                f"Data-loader state has {len(saved_order)} train samples, "
                f"but the current dataset has {0 if self.train_order is None else len(self.train_order)}"
            )
        if not np.array_equal(np.sort(saved_order), np.arange(len(saved_order))):
            raise ValueError('Invalid train_order in checkpoint')
        # A freshly loaded DataLoader is in canonical pickle order here.
        self.train_data[0] = np.take(self.train_data[0], saved_order, axis=0)
        self.train_data[1] = np.take(self.train_data[1], saved_order, axis=0)
        self.train_order = saved_order.copy()
        return True
    
    
    def get_eval_data(self):
        """Get eval data for next epoch
        """
        
        # reduce train data to a random subset
        if self.eval_data_reduction < 1.0:
            
            n = int(len(self.eval_data[0]) * self.eval_data_reduction)
            random_reduced_indices = generate_unique_randoms(count=n, min_value=0, max_value=len(self.eval_data[0])-1)
            X = np.take(a=self.eval_data[0], indices=random_reduced_indices, axis=0)
            y = np.take(a=self.eval_data[1], indices=random_reduced_indices, axis=0)
            p = np.take(a=self.eval_data[2], indices=random_reduced_indices, axis=0)
            r = np.take(a=self.eval_data[3], indices=random_reduced_indices, axis=0)
            s = np.take(a=self.eval_data[4], indices=random_reduced_indices, axis=0)
            
            self.current_eval_ids = [self.sample_ids['eval'][int(i)] for i in random_reduced_indices]
            return X, y, p, r, s
        
        # with no modifications
        else:
            
            self.current_eval_ids = self.sample_ids['eval']
            return self.eval_data[0], self.eval_data[1], self.eval_data[2], self.eval_data[3], self.eval_data[4]
        
    
    def get_test_data(self):
        """Get test data for next epoch
        """
        
        return self.test_data[0], self.test_data[1], self.test_data[2], self.test_data[3], self.test_data[4]
    
    
    def reorder_list(lst, indices):
        """Create a reordered copy of a given list
        """
        
        # create an empty list to store the elements in their new order
        tmp = []
        
        # iterate over the indices and add the corresponding elements from the original list to the new list
        for index in indices:
            
            if 0 <= index < len(lst):
                
                tmp.append(lst[index])
                
        return tmp
    
    
    def load_pkl_data(self, paths, type):
        """Load track and transformation data from a .pkl file
        """
        
        nX = []
        ny = []
        npos = []
        nrot = []
        nsrc = []
        
        # load data from source paths
        for p in paths:
        
            with open(p, 'rb') as f:
                ego_data = pickle.load(f)
                
            for id in ego_data:
                
                nX.append(ego_data[id]['X'])
                ny.append(ego_data[id]['y'][...,:2])
                npos.append(ego_data[id]['reference_position'])
                nrot.append(ego_data[id]['rotation_angle'])
                nsrc.append(ego_data[id]['source'])
                self.sample_keys[type].append(id)
                self.sample_ids[type].append(f"{type}:{ego_data[id]['source']}:{id}")
                
        return [np.array(nX), np.array(ny), np.array(npos), np.array(nrot), np.array(nsrc)]
