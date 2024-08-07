public_static_opts = {
    # graph preprocessing
    'self_loop': False,
    'udgraph': True,
    'lcc': True,
    
    # training
    'early_stop': True,
    'patience': 300,
    
    # model
    'n_hidden': 64,
    
    # 
    'n_cv': 1,
    'start_cv': 0
}