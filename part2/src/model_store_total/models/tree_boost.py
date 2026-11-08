from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

from .models_utils import tree_and_boosting_pipeline

# --- Model setup ------------------------------------------------------
TREE_PARAMS = {
    "max_depth": 8, 
    "min_samples_leaf": 20, 
    "random_state": 0
}

tree_model = DecisionTreeRegressor(**TREE_PARAMS) # type: ignore

BOOST_PARAMS = {
    "max_iter": 400, 
    "learning_rate": 0.05, 
    "max_depth": 4, 
    "min_samples_leaf": 20,
    "random_state": 0
}
boosting_model = HistGradientBoostingRegressor(**BOOST_PARAMS)

if __name__ == "__main__":
    tree_and_boosting_pipeline(tree_model, 'arvore')
    tree_and_boosting_pipeline(boosting_model, 'boosting')