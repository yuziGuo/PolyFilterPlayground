from data.citation_full_dataloader import  citation_full_supervised_loader
from data.coauthor_full_dataloader import  coauthor_full_supervised_loader
from data.geom_dataloader import geom_dataloader
from data.linkx_dataloader import linkx_dataloader
from data.platonov_dataloader import platonov_dataloader
import time

def build_dataset(args):
    if args.dataset in ['twitch-gamer', 'Penn94', 'genius']:
        loader = linkx_dataloader(args.dataset, 
                                  args.gpu, 
                                  args.self_loop, 
                                  n_cv=1)
    elif args.dataset in ['citeseerfull', 'pubmedfull', 'corafull']:
        loader = citation_full_supervised_loader(args.dataset, 
                                                 args.gpu, 
                                                 args.self_loop, 
                                                 n_cv=args.n_cv)
    elif args.dataset in ['cs', 'physics']:
        loader = coauthor_full_supervised_loader(args.dataset, 
                                                 args.gpu, 
                                                 args.self_loop, 
                                                 n_cv=args.n_cv)
    elif args.dataset.startswith('geom'):
        dataset = args.dataset.split('-')[1]
        loader = geom_dataloader(dataset, 
                                 args.gpu, 
                                 args.self_loop, 
                                 digraph=not args.udgraph, 
                                 n_cv=args.n_cv, 
                                 cv_id=args.start_cv
                                 )
    elif args.dataset in ['questions', 'roman-empire', 'minesweeper', 'tolokers', 
                          'amazon_ratings', 'chameleonF', 'squirrelF']:
        if args.dataset in ['chameleonF', 'squirrelF']:
            dataset = f'{args.dataset[:-1]}_filtered'
        else:
            dataset = args.dataset
        loader = platonov_dataloader(dataset, 
                                     args.gpu, 
                                     args.self_loop, 
                                     digraph=not args.udgraph, 
                                     n_cv=args.n_cv, 
                                     cv_id=args.start_cv
                                     )
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))    
    loader.load_data()
    # print('Sleep for 1 sec')
    # time.sleep(1)
    # To prevent 'Exception ignored in: <module 'threading' from '/home/yuhe_guo/miniconda3/lib/python3.9/threading.py'>'
    return loader