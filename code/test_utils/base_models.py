import time
import ConfigSpace as CS
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, make_scorer


def RF_search_space(seed=42):
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


def RFR_target_function(config, budget, **kwargs):
    # Extracting support information
    mse_scorer = make_scorer(mean_squared_error)
    seed = kwargs["seed"]
    train_X = kwargs["train_X"]
    train_y = kwargs["train_y"]
    test_X = kwargs["test_X"]
    test_y = kwargs["test_y"]
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

    test_mse = mse_scorer(model, test_X, test_y)

    result = {
        "fitness": valid_mse,  # DE/DEHB minimizes
        "cost": cost,
        "info": {
            "test_score": test_mse,
            "budget": budget
        }
    }

    return result


def RFC_target_function(config, budget, **kwargs):
    # Extracting support information
    accuracy_scorer = make_scorer(accuracy_score)
    seed = kwargs["seed"]
    train_X = kwargs["train_X"]
    train_y = kwargs["train_y"]
    test_X = kwargs["test_X"]
    test_y = kwargs["test_y"]
    valid_X = kwargs["valid_X"]
    valid_y = kwargs["valid_y"]
    max_budget = kwargs["max_budget"]

    if budget is None:
        budget = max_budget

    start = time.time()
    # Building model
    model = RandomForestClassifier(
        **config.get_dictionary(),
        n_estimators=int(budget),
        bootstrap=True,
        random_state=seed,
    )
    # Training the model on the complete training set
    model.fit(train_X, train_y)

    # Evaluating the model on the validation set
    valid_accuracy = accuracy_scorer(model, valid_X, valid_y)
    cost = time.time() - start

    test_accuracy = accuracy_scorer(model, test_X, test_y)

    result = {
        "fitness": 1 - valid_accuracy,  # DE/DEHB minimizes
        "cost": cost,
        "info": {
            "test_score": test_accuracy,
            "budget": budget
        }
    }

    return result
