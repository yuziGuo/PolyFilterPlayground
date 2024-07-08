public_hypers_default = {
    """
    The keys, for some models possibly together with some keys in `private_hypers`, 
    will be used in the following funtion `convert_dict_to_optuna_suggested`, 
    to produce a `suggestor`.
    """

    """
    The **keys** in this dictionary contains default hyperparameters for some models. 
    These keys, possibly combined with some in `private_hypers`, will be 
    used in the function `convert_dict_to_optuna_suggested` to produce a `suggestor`.
    """

    # Model structure
    'n_layers': 2,
    
    # Hypers for different parts of the model
    'lr1': 1e-2,
    'lr2': 1e-2,

    'wd1': 5e-4,
    'wd2': 5e-4,
    
    'dropout': 0.5,
    'dropout2': 0.5,
}


def convert_dict_to_optuna_suggested(dic, model):
    """
    Converts a dictionary of hyperparameters into a function that uses Optuna 
    to suggest new hyperparameter values.

    Args:
        dic (dict): Dictionary containing the default hyperparameters.
        model (str): The model name to determine specific cases for suggestion.

    Returns:
        function: A function that returns a set of hyperparameters suggested 
                  by Optuna each time it is called.
                  
    Optuna Suggestion Ranges:
        - learning rates:  [0.0005, 0.001, 0.005, 0.01, 0.02, ..., 0.05]
        - weight decays:  [1e-3, 1e-8, step=1e-1]
        - dropout rates: [0.0, 0.9, step=0.1]
        - layer nums: [4, 20, step=4]
    """

    def _specific_cases(trial, k):
        """
        Handles GPR-specific hyperparameter suggestions.
        """

        if model.startswith('GPRGNN'):
            if k == 'alpha':
                return trial.suggest_float(k, 0.1, 0.9, step=0.1)
        else:
            raise NotImplementedError(f"Specific case for {model} not implemented.")
    
    def suggest_args(trial):
        """
        Suggests hyperparameter values using Optuna.

        Args:
            trial (optuna.trial): Optuna trial object.

        Returns:
            dict: Dictionary of suggested hyperparameters.
        """
        args = dict()
        map_lr = {-0.02: 0.0005,  -0.01: 0.001, 0. : 0.005}

        for k in dic.keys():
            # Suggest learning rates
            if k.startswith('lr'):
                _lr = trial.suggest_float(k, -0.02, 0.05, step=0.01)
                if _lr <= 0:
                    _lr = map_lr[_lr]
                args[k] = round(_lr,4)

            # Suggest weight decays
            elif k.startswith('wd'):
                _wd = trial.suggest_int(k, -8, -3)
                args[k] = float('1e'+str(_wd))

            # Suggest dropouts
            elif k.startswith('dropout'):
                _dp = trial.suggest_float(k, 0.0, 0.9, step=0.1)
                args[k] = round(_dp, 4)

            # Suggest number of layers
            elif k.startswith('n_layers'):
                args[k] = trial.suggest_int(k, 4, 20, step=4)
                pass

            # Handle specific cases
            else:
                _v = _specific_cases(trial, k)
                args[k] = _v
                NotImplementedError
        return args
    
    return suggest_args
