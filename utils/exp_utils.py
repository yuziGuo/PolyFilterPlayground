import subprocess
import shutil
import json
import os
import time

def get_commit_id():
    try:
        # Run the git command to get the commit ID
        commit_id = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
        return commit_id
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return None


def get_branch_name():
    try:
        # Run the git command to get the branch name
        branch_name = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip().decode("utf-8")
        return branch_name
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return None


def _get_optuna_cache_dir(static_args, 
                          commit_id_abbr = None,
                          branch_name = None
                          ):
    opt_cache_base = 'cache/OptunaTrials'
    if commit_id_abbr is None:
        commit_id_abbr = get_commit_id()[:6]
    if branch_name is None:
        branch_name = get_branch_name()

    if static_args.study_kw == 'none':
        dir_name = f'{opt_cache_base}/{branch_name}-{commit_id_abbr}/'
    else:
        dir_name = f'{opt_cache_base}/studykw={static_args.study_kw}||br={branch_name}||cmt={commit_id_abbr}/'

    return dir_name


def _prepare_optuna_cache_dir(static_args):
    """
    ./cache/OptunaTrials/
    ---studykw=graphnorm=none||br={branch-name}||cmt={commit-id}/
    ------{model-A}-{dataset-1}.db
    ------{model-B}-{dataset-2}.db
    ------{model-C}-{dataset-3}.db

    ------zipped-opts|{model-A}|{dataset-1}|{MMDD-HH:MM}/
    ---------[the copied hyper files]
    ---------dumped_static_args.json

    ------zipped-opts|{model-A}|{dataset-1}|{MMDD-HH:MM}/
    ---------[the copied hyper files]
    ---------dumped_static_args.json

    """

    # Makedir
    opt_cache_base = 'cache/OptunaTrials'
    commit_id_abbr = get_commit_id()[:6]
    branch_name = get_branch_name()
    
    if static_args.study_kw == 'none':
        dir_name = f'{opt_cache_base}/{branch_name}-{commit_id_abbr}/'
        os.makedirs(dir_name, exist_ok=True)
    else:
        dir_name = f'{opt_cache_base}/studykw={static_args.study_kw}||br={branch_name}||cmt={commit_id_abbr}/'
        os.makedirs(dir_name, exist_ok=True)
    
    # Record options
    ## Prepare sub-directory with timestamp
    timestamp = time.strftime("%m%d-%H:%M", time.localtime())
    opts_rec_path = os.path.join(dir_name, 
                                 f'xerox-opts|{static_args.model}|{static_args.dataset}|{timestamp}/')
    os.makedirs(opts_rec_path, exist_ok=True)
    os.makedirs(opts_rec_path+'/src_copied', exist_ok=True)
    
    ## Copy tuning options
    opts_src_dir = 'opts/tune/'
    for _ in os.listdir(opts_src_dir):
        if os.path.isdir(os.path.join(opts_src_dir, _)):
            continue
        s = os.path.join(opts_src_dir, _)
        d = os.path.join(opts_rec_path+'/src_copied', _)
        shutil.copy2(s, d)
    
    with open(os.path.join(opts_rec_path, 'dumped_static_args.json'), 'w') as f:
        json.dump(vars(static_args), f, indent=4)

    return dir_name, opts_rec_path


def _record_tuning_opts_in_optuna_cache_dir(study, opts_rec_path):
    
    from optuna.distributions  import IntDistribution, FloatDistribution

    def _distribution_to_dict(d):
        # Assuming `d_` contains objects of type IntDistribution and FloatDistribution
        # Modify `d_` so that each distribution is represented as a dictionary
        if isinstance(d, (IntDistribution, FloatDistribution)):
            return {'type': type(d).__name__, 'high': d.high, 'low': d.low, 'log': d.log, 'step': d.step}
        return d

    with open(os.path.join(opts_rec_path, 'dumped_tuning_opts.json'), 'w') as f:
        d_ = study.trials[-1].distributions
        d_serializable = {k: _distribution_to_dict(v) for k, v in d_.items()}
        json.dump(d_serializable, f, indent=4)

    return

# test
if __name__ == '__main__':
    commit_id = get_commit_id()
    if commit_id:
        print(f"Current commit ID: {commit_id[:6]}")
    else:
        print("Failed to get commit ID.")

    branch_name = get_branch_name()
    if branch_name:
        print(f"Current branch name: {branch_name}")
    else:
        print("Failed to get branch name.")
