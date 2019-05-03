import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

class CategoricalColInt():
    """
    Tranforms column 'col_name' into n-1 categorical columns
    """
    def __init__(self, col_name, categorical_transforms, verbose=False):
        self.col_name = col_name
        self.categorical_transforms = categorical_transforms
        self.verbose = verbose
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy(deep=True)
        dummies = pd.get_dummies(X.loc[:, self.col_name], prefix=self.col_name)
        dummies = dummies.drop(dummies.columns[0], axis=1)
        all_dummies = pd.DataFrame(
            np.zeros((len(X), len(self.categorical_transforms[self.col_name]))),
            columns = self.categorical_transforms[self.col_name],
            index = dummies.index
        )
        all_dummies.loc[:, dummies.columns] = dummies.copy(deep=True)
        X = X.merge(all_dummies, left_index=True, right_index=True)
        X = X.drop(columns=self.col_name, axis=1)
        if self.verbose:
            print("Transforming {}. New cols: {}".format(self.col_name, self.categorical_transforms[self.col_name]))
            print("X shape: {}".format(X.shape))
        global transformed_features
        transformed_features = X
        return X


class SelectKBestFeatures():
    """
    Selects num of features to K_best
    """
    def __init__(self, k, verbose=False):
        self.k = k
        self.verbose = verbose
    
    def fit(self, X, y=None):
        self.selector = SelectKBest(score_func=f_classif, k=self.k).fit(X, y)
        return self

    def transform(self, X):
        X = X.copy(deep=True)
        self.selector.transform(X)
        df_scores = pd.DataFrame(X.columns, columns = ["col_name"])
        df_scores.loc[:, "score"] = self.selector.scores_
        df_scores = df_scores.sort_values(by="score", ascending=False)
        best_features = df_scores.head(self.k).loc[:, "col_name"]
        X = X.loc[:, best_features]
        if self.verbose:
            print("SelectKBest features selected: {}".format(best_features))
        global transformed_features
        transformed_features = X
        return X
    
    def set_params(self, k):
        self.k = k