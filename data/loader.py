# from dgl import ops
import torch as th
import random

class loader():
    def __init__(self, ds_name='Cora', 
                        self_loop=True,
                        process_features=False,
                        cross_validation=False,
                        largest_component=False,
                        n_cv=-1,
                        cv_id=-1,
                        ):
        self.g = None
        self.feature = None
        self.labels = None

        self.train_mask = None
        self.val_mask = None
        self.test_mask = None

        self.n_classes = None
        self.n_edges = None
        self.n_nodes = None

        self.cross_validation = cross_validation
        self.n_cv = n_cv
        self.cv_id = cv_id
        if cross_validation and self.n_cv > 1:
            self.cv_id = cv_id
        else:
            cv_id = 0
        
        self.ds_name = ds_name
        self.self_loop = self_loop
        self.largest_component = largest_component

    def set_split_seeds(self):
        self.seeds = [random.randint(0,10000) for i in range(self.n_cv)]

    def load_mask(self,p=None):
        if self.cross_validation:
            self.load_a_mask(p)
            self.cv_id += 1
        else:
            # For some datasets, mask is already loaded before.
            # print('Mask is already loaded!')
            pass

        return


    def _get_lcc(self):
        '''
        Return:
            True, if there exists over one connected components(cc), 
                and the largest cc is stored in two convenient Tensors
                i.e. `self.lcc_flags' and `lcc_map' .
            or False, if there exists only one connected component.
        '''
        from scipy.sparse import coo_array
        from scipy.sparse.csgraph import connected_components
        from collections import Counter
        from torch_geometric.utils.num_nodes import maybe_num_nodes
        import numpy as np

        edge_index = self.edge_index
        n = maybe_num_nodes(edge_index)
        m = edge_index.shape[-1]
        fill = np.ones(m)
        arr = coo_array((fill, 
                        (edge_index[0].detach().cpu().numpy(), 
                         edge_index[1].detach().cpu().numpy()
                        )),shape=(n, n))
        
        n_components, labels = connected_components(csgraph=arr, directed=False, return_labels=True)

        print(f'n_components: {n_components}')
        if n_components == 1:
            print(f"[INFO - dataloader] There is only one largest component!")
            self.largest_component = False
            return False
        # else: n_components > 1


        '''
        Below, we maintain two data structures to convert 
        node ids **in the original graph** 
        to node ids **in the lcc subgraph**.
        
        - `self.lcc_flags`   shape: (n,)
            Positions   0       1       2       3         n-1   
            Values      [False, True,   False,  True ..., True]

        - `self.lcc_map`   shape: (n,)
            Positions   0       1       2       3         n-1   
            Values      [-1,    0,      -1,     1,  ..., _n-1]

        
        Remark: These two data structures bring convenience to 
        transferring node ids! The usages can be found in 
        `self.__filter_edge_index`.
        
        '''
        lcc_id, n_ = Counter(labels).most_common(1)[0]
        self.lcc_flags = th.tensor((labels==lcc_id)).to(self.device) # (n,)

        self.lcc_map = - th.ones(n, dtype=int, device=self.device)
        self.relabeled_nids = th.arange(n_, device=self.device)
        self.lcc_map[self.lcc_flags] = self.relabeled_nids  #(n,)

        return True


    def _filter_edge_index(self):
        e = self.edge_index.T #(m,2)

        # Retain edges where both endpoints are 
        # in the largest connected component (LCC)
        t = self.lcc_flags[e]
        filter = t[:,0] | t[:,1]              # (m,)
        edge_index_filtered = e[filter,:].T   # (m_,2)

        # Reindex edge_index to new indices 
        # in the LCC subgraph
        edge_index_filtered_reindexed = self.lcc_map[edge_index_filtered]

        return edge_index_filtered_reindexed 

    def _filter_attrs(self):
        self.features = self.features[self.lcc_flags,:]
        self.labels = self.labels[self.lcc_flags]
        self.n_edges = self.edge_index.shape[-1]
        self.n_nodes = self.edge_index.max().item() + 1


    def load_data(self):
        self.load_vanilla_data()
        if self.largest_component:
            if self._get_lcc():  # Return False is there is only one cc
                self.edge_index = self._filter_edge_index()
                self._filter_attrs()


    def load_vanilla_data(self):
        '''
            Implemented in child-classes
            load features and labels
        '''
        pass

    def process_features():
        pass

    def process_graph(self):
        self.g = self.g.remove_self_loop()
        if self.self_loop:
            self.g = self.g.add_self_loop()
        self.n_edges = self.g.number_of_edges()


