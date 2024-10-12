from data import *
import time

def build_dataset(args):
    if args.gpu < 0:
        dargs = vars(args)
        dargs.update({'gpu':'cpu'})
    
    # match a loader function
    if args.dataset in ['twitch-gamer', 'Penn94', 'genius']:
        loader = linkx_dataloader
    elif args.dataset in ['citeseerfull', 'pubmedfull', 'corafull', 
                          'citeseer','cora','pubmed']:
        loader = citation_full_supervised_loader
    elif args.dataset in ['photo', 'computers']:
        loader = amazon_dataloader
    elif args.dataset.startswith('geom'):
        args.dataset = args.dataset.split('-')[1]
        loader = geom_dataloader
    elif args.dataset in ['questions', 'roman-empire', 'minesweeper', 'tolokers', 
                          'amazon_ratings', 'chameleonF', 'squirrelF']:
        if args.dataset in ['chameleonF', 'squirrelF']:
            args.dataset = f'{args.dataset[:-1]}_filtered'
        else:
            args.dataset = args.dataset
        loader = platonov_dataloader
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))    
    
    # build the loader
    loader = loader(args.dataset, 
                    args.gpu, 
                    args.self_loop, 
                    digraph=not args.udgraph, 
                    largest_component=args.lcc,
                    n_cv=args.n_cv, 
                    cv_id=args.start_cv
                    )
    loader.load_data()

    # Prevent an exception: 'Exception ignored in: <module 'threading' from '/home/yuhe_guo/miniconda3/lib/python3.9/threading.py'>'
    print('[INFO - dataloader] Sleep for 1 sec')
    time.sleep(1)
    
    return loader


if __name__ == '__main__':
    import argparse
    for ds_name in ['Penn94', 'citeseerfull', 'questions']:
        args = argparse.Namespace(
            gpu=-1,
            dataset=ds_name,
            self_loop=True,
            udgraph=False,
            lcc=True,
            n_cv=5,
            start_cv=0
        )
        loader=build_dataset(args)
        print(loader)