import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import math
import patchifier

def point_in_patch(point, patch):
    patch_x_min, patch_x_max, patch_y_min, patch_y_max = patch
    point_x, point_y = point

    if (patch_x_min <= point_x <= patch_x_max) and (patch_y_min <= point_y <= patch_y_max):
        return True
    else:
        return False

def _check_array(array):
    if isinstance(array, pd.DataFrame):
        return array.values
    elif isinstance(array, pd.Series):
        return array.values
    elif isinstance(array, list):
        return np.array(list)
    elif isinstance(array, np.ndarray):
        return array
    else:
        raise TypeError('Input array must be a pandas.DataFrame, a numpy.ndarray or a list.')

def _check_predictor(predictor):
    fitted = False
    for k, v in predictor.__dict__.items():
        if k[-1] == '_' and v:
            fitted = True
            break

    return fitted


def _patches_overlap(patcha, patchb):
    patcha_x_min, patcha_x_max, patcha_y_min, patcha_y_max = patcha
    patchb_x_min, patchb_x_max, patchb_y_min, patchb_y_max = patchb

    if (patcha_x_min <= patchb_x_min <= patcha_x_max) or (patcha_x_min <= patchb_x_max <= patcha_x_max):
        if (patcha_y_min <= patchb_y_min <= patcha_y_max) or (patcha_y_min <= patchb_y_max <= patcha_y_max):
            return True

    return False


def _merge_patches(patcha, patchb):

    patcha_x_min, patcha_x_max, patcha_y_min, patcha_y_max = patcha
    patchb_x_min, patchb_x_max, patchb_y_min, patchb_y_max = patchb

    new_x_min = min(patcha_x_min, patchb_x_min)
    new_x_max = max(patcha_x_max, patchb_x_max)
    new_y_min = min(patcha_y_min, patchb_y_min)
    new_y_max = max(patcha_y_max, patchb_y_max)

    return (new_x_min, new_x_max, new_y_min, new_y_max)

class BOVW(object):

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

    def convert_dictionary_predictions_to_counts(self, predictions, density=1.0):
        counts, bins = np.histogram(predictions, bins=self.dictionary_model.n_clusters, density=density)
        return counts

    def predict_dictionary_and_convert_to_counts(self, X):
        X = _check_array(X)
        if X.shape[0] == 0:
            return np.zeros(shape=(self.dictionary_model.n_clusters,))
        
        preds = self.predict_dictionary(X)
        return self.convert_dictionary_predictions_to_counts(preds)

    def predict(self, X):
        X = _check_array(X)
        preds = self.classifier_model.predict(X)
        return preds

    def score(self, X, y):
        X = _check_array(X)
        y = _check_array(y)
        sc = self.classifier_model.score(X, y)
        return sc


    def detect_object_in_image(self,
                               keypoint_list,
                               descriptor_list,
                               image_shape,
                               min_patch_size,
                               max_patch_size,
                               scaling_factor,
                               x_overlap,
                               y_overlap):
        # make sure we have trained the classifier model
        if not _check_predictor(self.classifier_model):
            raise ValueError("You need to train a classifier model before detecting objects")


        multiscale_patch_array = patchifier.make_multiscale_patch_array(image_shape,
                                                                        max_patch_size,
                                                                        min_patch_size,
                                                                        scaling_factor,
                                                                        x_overlap,
                                                                        y_overlap)



        # for each patch, get all the keypoints in the patch, then convert
        # the descriptors to counts and run the classifier
        object_locations = []
        for patch_scale in multiscale_patch_array:
            for patch in patch_scale:
                patch_x_min = patch[0]
                patch_x_max = patch[1]
                patch_y_min = patch[2]
                patch_y_max = patch[3]


                # print(patch_x_min)
                # print(patch_y_min)
                # get the keypoints within the patch
                kps_idx_in_range = [idx for idx, kp in enumerate(keypoint_list)
                                if (patch_x_min <= kp[1] <= patch_x_max) and
                                   (patch_y_min <= kp[0] <= patch_y_max)]

                if len(kps_idx_in_range) == 0: continue

                patch_descriptors = [descriptor_list[idx] for idx in kps_idx_in_range]
                patch_descriptors = np.vstack(tuple(patch_descriptors))
                patch_hist = self.predict_dictionary_and_convert_to_counts(patch_descriptors)
                prediction = self.classifier_model.predict(patch_hist.reshape(1,-1))
                object_locations.append((prediction[0], patch))

        return object_locations


    def consolidate_object_locations(self,
                                     object_location_list,
                                     class_to_consolidate):

        patches_of_interest = [patch for pc, patch in object_location_list if pc == class_to_consolidate]
        patches_of_interest.sort(key= lambda x: x[0])

        distinct_locations = []
        last_patches_of_interest_length = len(patches_of_interest) + 1
        idxs_to_pop_len = 1e6

        while idxs_to_pop_len > 0:
            cur_patch = patches_of_interest.pop()
            # get and merge patches
            idxs_to_pop = set()
            for idx, patch in enumerate(patches_of_interest):
                if _patches_overlap(cur_patch, patch):
                    cur_patch = _merge_patches(cur_patch, patch)
                    idxs_to_pop.add(idx)

            for idx in idxs_to_pop:
                patches_of_interest = [patch for idx, patch in enumerate(patches_of_interest) if idx not in idxs_to_pop]

            patches_of_interest.append(cur_patch)
            patches_of_interest.sort(key= lambda x: x[0])
            idxs_to_pop_len = len(idxs_to_pop)

        distinct_locations = patches_of_interest
        return distinct_locations
