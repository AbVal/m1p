import os
import time
import numpy as np
import pandas as pd
import ConfigSpace as CS
import seaborn as sns
import matplotlib.pyplot as plt
from dehb import DEHB
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, make_scorer

import warnings
warnings.filterwarnings('ignore')


def constraints_setup(min_budget=2, max_budget=150):
    seed = 123
    np.random.seed(seed)
    return min_budget, max_budget


def create_search_space(seed=123):
    """Parameter space to be optimized --- contains the hyperparameters
    """
    cs = CS.ConfigurationSpace(seed=seed)

    cs.add_hyperparameters([
        CS.UniformIntegerHyperparameter(
            'max_depth', lower=1, upper=20, default_value=2, log=False
        ),
        CS.UniformIntegerHyperparameter(
            'min_samples_split', lower=2, upper=256, default_value=2, log=True
        ),
        CS.UniformFloatHyperparameter(
            'max_features', lower=0.1, upper=0.9, default_value=0.5, log=False
        ),
        CS.UniformIntegerHyperparameter(
            'min_samples_leaf', lower=1, upper=256, default_value=1, log=True
        ),
    ])
    return cs


def prepare_cali_housing(seed=123):
    _data = fetch_california_housing(as_frame=True)

    train_X, rest_X, train_y, rest_y = train_test_split(
      _data.get("data")[:200],
      np.log(_data.get("target")[:200]),
      train_size=0.7,
      shuffle=True,
      random_state=seed
    )

    # 10% test and 20% validation data
    valid_X, test_X, valid_y, test_y = train_test_split(
      rest_X, rest_y,
      test_size=0.3333,
      shuffle=True,
      random_state=seed
    )

    return train_X, train_y, valid_X, valid_y, test_X, test_y, 'Cali Housing'


def cali_target_function(config, budget, **kwargs):
    # Extracting support information
    mse_scorer = make_scorer(mean_squared_error)
    seed = kwargs["seed"]
    train_X = kwargs["train_X"]
    train_y = kwargs["train_y"]
    valid_X = kwargs["valid_X"]
    valid_y = kwargs["valid_y"]
    max_budget = kwargs["max_budget"]

    if budget is None:
        budget = max_budget

    start = time.time()
    # Building model 
    model = RandomForestRegressor(
        **config.get_dictionary(),
        n_estimators=int(budget),
        bootstrap=True,
        random_state=seed,
    )
    # Training the model on the complete training set
    model.fit(train_X, train_y)

    # Evaluating the model on the validation set
    valid_mse = mse_scorer(model, valid_X, valid_y)
    cost = time.time() - start

    result = {
        "fitness": valid_mse,  # DE/DEHB minimizes
        "cost": cost,
        "info": {
            "budget": budget
        }
    }
    return result


def run_dehb_test(dehb, train_X, train_y, valid_X, valid_y, runs=10, fevals=100, seed=123):
    total_df = pd.DataFrame(columns=['score', 'feval', 'strategy'])

    for i in range(runs):
        # Resetting to begin optimization again
        dehb.reset()
        # Executing a run of DEHB optimization
        trajectory, runtime, history = dehb.run(
            fevals=fevals,
            verbose=False,
            save_intermediate=False,
            seed=seed,
            train_X=train_X,
            train_y=train_y,
            valid_X=valid_X,
            valid_y=valid_y,
            max_budget=dehb.max_budget
        )
        trajectory_df = pd.DataFrame({'score': trajectory,
                                      'feval': np.arange(1, len(trajectory) + 1),
                                      'strategy': [dehb.strategy] * len(trajectory)})
        total_df = pd.concat([total_df, trajectory_df])
    return total_df


def plot_experiments(df_list, fig_name=None, fig_title=None):
    df = pd.concat(df_list)
    
    if len(df_list) > 1:
        sns.relplot(df, x='feval', y='score', hue='strategy', kind='line')
    else:
        sns.relplot(df, x='feval', y='score', kind='line')
    plt.grid()
    plt.title(fig_title)
    if fig_name is not None:
        plt.savefig(os.path.join('../figures', fig_name), bbox_inches='tight')
    return plt.gca()


def run_basic_experiment(fig_name=None, fig_title=None, df_name=None, fevals=75, runs=10):
    min_budget, max_budget = constraints_setup()
    cs = create_search_space()
    dimensions = len(cs.get_hyperparameters())
    print(f"Dimensionality of search space: {dimensions}")
    train_X, train_y, valid_X, valid_y, test_X, test_y, dataset = \
        prepare_cali_housing()

    print(dataset)
    print("Train size: {}\nValid size: {}\nTest size: {}".format(
        train_X.shape, valid_X.shape, test_X.shape
    ))

    dehb_rand1 = DEHB(
                    f=cali_target_function,
                    cs=cs,
                    dimensions=dimensions,
                    min_budget=min_budget,
                    max_budget=max_budget,
                    n_workers=1,
                    output_path="./rand1_bin_logs",
                    strategy='rand1_bin',
    )
    df_rand1 = run_dehb_test(dehb_rand1, train_X, train_y, valid_X, valid_y, fevals=fevals, runs=runs)
    if df_name is not None:
        df_rand1.to_csv(df_name, index=False)
    return df_rand1, plot_experiments([df_rand1], fig_name=fig_name, fig_title=fig_title)


def run_noise_experiment(fig_name=None, fig_title=None, df_prefix=None, noise_scale=0.5, fevals=75, runs=10):
    min_budget, max_budget = constraints_setup()
    cs = create_search_space()
    dimensions = len(cs.get_hyperparameters())
    print(f"Dimensionality of search space: {dimensions}")
    train_X, train_y, valid_X, valid_y, test_X, test_y, dataset = \
        prepare_cali_housing()

    print(dataset)
    print("Train size: {}\nValid size: {}\nTest size: {}".format(
        train_X.shape, valid_X.shape, test_X.shape
    ))

    dehb_rand1 = DEHB(
                    f=cali_target_function,
                    cs=cs,
                    dimensions=dimensions,
                    min_budget=min_budget,
                    max_budget=max_budget,
                    n_workers=1,
                    output_path="./rand1_bin_logs",
                    strategy='rand1_bin',
    )
    df_rand1 = run_dehb_test(dehb_rand1, train_X, train_y, valid_X, valid_y, fevals=fevals, runs=runs)

    dehb_noisy1 = DEHB(
                    f=cali_target_function, 
                    cs=cs, 
                    dimensions=dimensions, 
                    min_budget=min_budget, 
                    max_budget=max_budget,
                    n_workers=1,
                    output_path="./noisy1_bin_logs",
                    strategy='noisy1_bin',
                    noise_scale=noise_scale
    )
    df_noisy1 = run_dehb_test(dehb_noisy1, train_X, train_y, valid_X, valid_y, fevals=fevals, runs=runs)

    if df_prefix is not None:
        df_rand1.to_csv(df_prefix + '_rand1.csv', index=False)
        df_noisy1.to_csv(df_prefix + '_noisy1.csv', index=False)

    return df_rand1, df_noisy1, plot_experiments([df_rand1, df_noisy1], fig_name=fig_name, fig_title=fig_title)
    