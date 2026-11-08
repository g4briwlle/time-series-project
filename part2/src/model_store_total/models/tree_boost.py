from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

from .models_utils import tree_and_boosting_pipeline, tree_and_boosting_recursive_pipeline

# --- Model setup ------------------------------------------------------
TREE_PARAMS = {
    "max_depth": 8, 
    "min_samples_leaf": 20, 
    "random_state": 0
}

tree_model_direct = DecisionTreeRegressor(**TREE_PARAMS) # type: ignore
tree_model_recursive = DecisionTreeRegressor(**TREE_PARAMS) # type: ignore

BOOST_PARAMS_DIRECT = {
    "max_iter": 400, 
    "learning_rate": 0.05, 
    "max_depth": 4, 
    "min_samples_leaf": 20,
    "random_state": 0
}
BOOST_PARAMS_RECURSIVE = BOOST_PARAMS_DIRECT.copy()
BOOST_PARAMS_RECURSIVE["l2_regularization"] = 1.5

boosting_model_direct = HistGradientBoostingRegressor(**BOOST_PARAMS_DIRECT)
boosting_model_recursive = HistGradientBoostingRegressor(**BOOST_PARAMS_RECURSIVE)

if __name__ == "__main__":
    tree_and_boosting_pipeline(tree_model_direct, 'arvore_direto')
    tree_and_boosting_pipeline(boosting_model_direct, 'boosting_direto')
    tree_and_boosting_recursive_pipeline(tree_model_recursive, 'arvore_recursivo')
    tree_and_boosting_recursive_pipeline(boosting_model_recursive, 'boosting_recursivo')