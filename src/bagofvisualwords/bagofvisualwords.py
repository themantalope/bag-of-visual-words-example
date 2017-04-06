import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import math


def _check_array(array):
    if isinstance(array, pd.DataFrame):
        return array.values
    elif isinstance(array, list):
        return np.array(list)
    elif isinstance(array, np.ndarray):
        return array
    else:
        raise TypeError('Input array must be a pandas.DataFrame, a numpy.ndarray or a list.')



class BOVW(object):

    def __init__(self):
        self.something = None

    def fit(self, X, y, model=ExtraTreesClassifier()):
        X = _check_array(X)
        y = _check_array(y)
        self.classifier_model = model
        self.classifier_model.fit(X, y)

    def cross_validate_classifier_model(self,
                                        X,
                                        y,
                                        model=ExtraTreesClassifier(),
                                        search_params={"n_estimators":[int(math.floor(x)) for x in np.logspace(1, 3, num=5)],
                                                       "max_features":["auto", 'sqrt', 'log2', None],
                                                       "max_depth":[int(math.floor(x)) for x in np.logspace(0, 2, num=5)]},
                                        verbose=0,
                                        cross_validator=StratifiedKFold(n_splits=5),n_jobs=-1):

        X = _check_array(X)
        y = _check_array(y)
        gscv = GridSearchCV(model,
                            param_grid=search_params,
                            verbose=verbose,
                            cv=cross_validator,
                            n_jobs=n_jobs)
        gscv.fit(X, y)
        return gscv


    def fit_dictionary(self, X, model=KMeans(n_clusters=100)):
        X = _check_array(X)
        self.dictionary_model = model
        self.dictionary_model.fit(X=X)


    def predict_dictionary(self, X):
        X = _check_array(X)
        preds = self.dictionary_model.predict(X)
        return preds

    def convert_dictionary_predictions_to_counts(self, predictions):
        counts, bins = np.histogram(predictions, bins=self.dictionary_model.n_clusters, density=1.0)
        return counts

    def predict_dictionary_and_convert_to_counts(self, X):
        X = _check_array(X)
        preds = self.predict_dictionary(X)
        return self.convert_dictionary_predictions_to_counts(preds)

    def predict(self, X):
        X = _check_array(X)
        preds = self.classifier_model.predict(X)
        return preds
