from data.split import index_to_mask
from data.loader import loader

from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
import numpy as np
import torch as th

# from utils.data_split import random_planetoid_splits
import os


class platonov_dataloader(loader):
    def __init__(self, ds_name, device='cuda:0', self_loop=True, 
                    digraph=False, n_cv=3, cv_id=0,
                    needs_edge=False):
        super(platonov_dataloader, self).__init__(
            ds_name, 
            cross_validation=True, 
            n_cv=n_cv, 
            cv_id=cv_id,
            needs_edge=needs_edge
            )
        self.device = device
        self.digraph = digraph
        self.self_loop = self_loop

    def load_vanilla_data(self):
        dataset = np.load(os.path.join('dataset/platonov', f'{self.ds_name.replace("-", "_")}.npz'))
        self.edge_index = th.LongTensor(dataset['edges']).to(self.device).T
        if not self.digraph:
            self.edge_index = to_undirected(self.edge_index)
        if self.self_loop:
            self.edge_index = remove_self_loops(self.edge_index)[0]
            self.edge_index = add_self_loops(self.edge_index)[0]
        self.features = th.FloatTensor(dataset['node_features']).to(self.device)
        self.labels = th.LongTensor(dataset['node_labels']).to(self.device)
        if self.labels.dim()==2 and self.labels.shape[-1]==1:
            self.labels = self.labels.squeeze()
        self.n_nodes = self.labels.shape[0]
        self.in_feats = self.features.shape[1]
        
        # infer the number of classes for non one-hot and one-hot labels
        if len(self.labels.shape) == 1:
            labels = self.labels.unsqueeze(1)
        self.n_classes = max(self.labels.max().item() + 1, labels.shape[-1])
        self.n_edges = self.edge_index.shape[-1]


    def load_fixed_splits(self):
        dataset = np.load(os.path.join('dataset/platonov', f'{self.ds_name.replace("-", "_")}.npz'))
        self.train_mask = th.tensor(dataset['train_masks'][self.cv_id])
        self.val_mask = th.tensor(dataset['val_masks'][self.cv_id])
        self.test_mask = th.tensor(dataset['test_masks'][self.cv_id])
        return 

    def load_a_mask(self, p=None):
        self.load_fixed_splits()
        return 

        

def test_platonov():
    loader = platonov_dataloader('questions', 'cuda:1', True)
    loader.load_data()
    loader.load_mask()
    print('Success!')


if __name__=='__main__':
    test_platonov()
    
