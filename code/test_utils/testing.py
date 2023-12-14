import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dehb import DEHB
import warnings
warnings.filterwarnings('ignore')


def set_seed(seed=42):
    np.random.seed(seed)
    return seed


def run_dehb_test(dehb, dataset, runs=10, fevals=100, seed=42, hue='strategy', hue_name=None):
    total_df = pd.DataFrame(columns=['score', 'feval', hue])
    test_score = 0

    train_X, train_y, valid_X, valid_y, test_X, test_y = dataset['data']

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
            test_X=test_X,
            test_y=test_y,
            valid_X=valid_X,
            valid_y=valid_y,
            max_budget=dehb.max_budget
        )

        __hue_name = hue_name
        if hue_name is None:
            __hue_name = dehb.strategy

        trajectory_df = pd.DataFrame({'loss': trajectory,
                                      'feval': np.arange(1, len(trajectory) + 1),
                                      hue: [__hue_name] * len(trajectory)})
        total_df = pd.concat([total_df, trajectory_df])

        config, score, cost, budget, _info = history[-1]

        test_score += _info['test_score']

    test_score /= runs

    return total_df, test_score


def plot_experiments(df_list, fig_name=None, fig_title=None, hue='strategy'):
    df = pd.concat(df_list)
    df = df[df['feval'] >= 5]

    if len(df_list) > 1:
        sns.relplot(df, x='feval', y='loss', hue=hue, kind='line')
    else:
        sns.relplot(df, x='feval', y='loss', kind='line')
    plt.grid()
    plt.title(fig_title)
    if fig_name is not None:
        plt.savefig(os.path.join('../figures', fig_name), bbox_inches='tight')
    return plt.gca()


def run_basic_experiment(model, dataset,
                         fig_name=None, fig_title=None, df_name=None,
                         fevals=75, runs=10, seed=42):
    set_seed(seed=seed)

    dimensions = len(model.cs.get_hyperparameters())
    print(f"Dimensionality of search space: {dimensions}")

    data_name = dataset['name']
    print(data_name)

    train_X, train_y, valid_X, valid_y, test_X, test_y = dataset['data']
    print("Train size: {}\nValid size: {}\nTest size: {}".format(
        train_X.shape, valid_X.shape, test_X.shape
    ))

    run, avg_test_score = run_dehb_test(model, dataset, fevals=fevals, runs=runs, seed=seed)
    if df_name is not None:
        run.to_csv(df_name, index=False)
    return run, avg_test_score, plot_experiments([run], fig_name=fig_name, fig_title=fig_title)


def run_comparison_experiment(model1, model2, dataset,
                              fig_name=None, fig_title=None, df_prefix=None,
                              fevals=75, runs=10, seed=42):
    set_seed(seed=seed)

    assert model1.cs == model2.cs, 'Search spaces differ'

    dimensions = len(model1.cs.get_hyperparameters())
    print(f"Dimensionality of search space: {dimensions}")

    data_name = dataset['name']
    print(data_name)

    train_X, train_y, valid_X, valid_y, test_X, test_y = dataset['data']
    print("Train size: {}\nValid size: {}\nTest size: {}".format(
        train_X.shape, valid_X.shape, test_X.shape
    ))

    run1, avg_test_score_1 = run_dehb_test(model1, dataset, fevals=fevals, runs=runs, seed=seed)

    run2, avg_test_score_2 = run_dehb_test(model2, dataset, fevals=fevals, runs=runs, seed=seed)

    if df_prefix is not None:
        run1.to_csv(df_prefix + '_1.csv', index=False)
        run2.to_csv(df_prefix + '_2.csv', index=False)

    return run1, avg_test_score_1, run2, avg_test_score_2, \
        plot_experiments([run1, run2], fig_name=fig_name, fig_title=fig_title)


def run_noise_ablation(noise_scales, dataset, dehb_params,
                       fig_name=None, fig_title=None,
                       fevals=75, runs=10):

    data_name = dataset['name']
    print(data_name)

    train_X, train_y, valid_X, valid_y, test_X, test_y = dataset['data']
    print("Train size: {}\nValid size: {}\nTest size: {}".format(
        train_X.shape, valid_X.shape, test_X.shape
    ))

    run_list = []
    test_scores = []

    for noise in noise_scales:

        model = DEHB(basic_noise_scale=noise, **dehb_params)
        run, avg_test_score = run_dehb_test(model, dataset, fevals=fevals, runs=runs,
                                            hue='base_noise_scale', hue_name=str(noise))
        run_list.append(run)
        test_scores.append(avg_test_score)

    return run_list, test_scores, plot_experiments(run_list, fig_name=fig_name,
                                                   fig_title=fig_title, hue='base_noise_scale')


def run_step_ablation(noise_steps, dataset, dehb_params,
                      fig_name=None, fig_title=None,
                      fevals=75, runs=10):

    data_name = dataset['name']
    print(data_name)

    train_X, train_y, valid_X, valid_y, test_X, test_y = dataset['data']
    print("Train size: {}\nValid size: {}\nTest size: {}".format(
        train_X.shape, valid_X.shape, test_X.shape
    ))

    run_list = []
    test_scores = []

    for step in noise_steps:

        model = DEHB(noise_step=step, **dehb_params)
        run, avg_test_score = run_dehb_test(model, dataset, fevals=fevals, runs=runs,
                                            hue='noise_step', hue_name=str(step))
        run_list.append(run)
        test_scores.append(avg_test_score)

    return run_list, test_scores, plot_experiments(run_list, fig_name=fig_name,
                                                   fig_title=fig_title, hue='noise_step')
