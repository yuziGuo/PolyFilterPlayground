public_hypers_default = {
    # structure
    'n_layers': 2,
    
    # within stacked layers
    'lr1': 1e-2,
    'lr2': 1e-2,
    'wd1': 5e-4,
    'wd2': 5e-4,
    'dropout': 0.5,
    'dropout2': 0.5,
}


def convert_dict_to_optuna_suggested(dic, model):
    def _specific_cases(trial, k):
        if model.startswith('GPRGNN'):
            if k == 'alpha':
                v = trial.suggest_float(k, 0.1, 0.9, step=0.1)
        else:
            NotImplementedError
        return v
    
    def suggest_args(trial):
        args = dict()
        map_lr = {-0.02: 0.0005,  -0.01: 0.001, 0. : 0.005}
        for k in dic.keys():
            # learning rates
            if k.startswith('lr'):
                _lr = trial.suggest_float(k, -0.02, 0.05, step=0.01)
                if _lr <= 0:
                    _lr = map_lr[_lr]
                args[k] = round(_lr,4)
            # weight decays
            elif k.startswith('wd'):
                _wd = trial.suggest_int(k, -8, -3)
                args[k] = float('1e'+str(_wd))
            # dropouts
            elif k.startswith('dropout'):
                _dp = trial.suggest_float(k, 0.0, 0.9, step=0.1)
                args[k] = round(_dp,4)
            # n_layers
            elif k.startswith('n_layers'):
                # args[k] = trial.suggest_int(k, 4, 20, step=4)
                args[k] = trial.suggest_int(k, 10, 10, step=4)
                pass
            else:
                _v = _specific_cases(trial, k)
                args[k] = _v
                NotImplementedError
        return args
    
    return suggest_args
