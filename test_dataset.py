from data.platonov_dataloader import platonov_dataloader
from data.geom_dataloader import geom_dataloader
from data.linkx_dataloader import linkx_dataloader

def test_platonov():
    ds_names = ['questions', 'roman-empire', 'minesweeper', 'tolokers', 'amazon_ratings']
    for ds in ds_names:
        loader = platonov_dataloader(ds, 'cuda:1', True)
        loader.load_data()
        loader.load_a_mask()
        print('Success!') 

        print(ds)
        print(f"features.shape: {loader.features.shape}")
        print(f"labels.shape: {loader.labels.shape}")
        print(f"n_classes: {loader.n_classes}")
        print(f"number of nodes in each class:")
        for c in range(loader.n_classes):
            print((loader.labels==c).sum())
        print()

def test_linkx():
    ds_names = ['genius']
    for ds in ds_names:
        loader = linkx_dataloader(ds, 'cuda:1', True)
        loader.load_data()
        loader.load_a_mask()
        print('Success!') 

        print(ds)
        print(f"features.shape: {loader.features.shape}")
        print(f"labels.shape: {loader.labels.shape}")
        print(f"n_classes: {loader.n_classes}")
        print(f"number of nodes in each class:")
        for c in range(loader.n_classes):
            print((loader.labels==c).sum())
        print()


def test_geom():
    loader = geom_dataloader('chameleon', 'cuda:1', False)
    loader.load_data()

if __name__=='__main__':
    test_platonov()