import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from copy import deepcopy


class iFusion:

    def __init__(self,
                 cv_num=5,
                 indiv_learner=LinearRegression(),
                 fusion_learner=LinearRegression()
                ):
        self.cv_num = cv_num
        self.indiv_learner = indiv_learner
        self.fusion_learner = fusion_learner


    def _load_data(self, X, y, id, indiv_num):
        self.X = X
        self.y = y
        self.id = id
        if indiv_num:
            self.indiv_num = indiv_num
        else:
            self.indiv_num = len(np.unique(id))


    def _fit_indiv_learners(self):
        self.indiv_learners = []
        for k in range(self.indiv_num):
            indiv_learner_copy = deepcopy(self.indiv_learner)
            indiv_learner_copy.fit(self.X[self.id == k], self.y[self.id == k])
            self.indiv_learners.append(indiv_learner_copy)


    def _predict_using_indiv_models(self):
        mask = id == self.target
        X_target, y_target = X[mask], y[mask]
        indiv_preds = []
        for k in range(self.indiv_num):
            if k == self.target:
                tmp = y_target.copy()
                kf = KFold(n_splits=self.cv_num)
                kf_split = kf.split(X_target, y_target)
                for train_index, test_index in kf_split:
                    indiv_learner = deepcopy(self.indiv_learner)
                    indiv_learner.fit(X_target[train_index], y_target[train_index])
                    tmp[test_index] = indiv_learner.predict(X_target[test_index])
                indiv_preds.append(tmp)
            else:
                indiv_preds.append(self.indiv_learners[k].predict(X_target))
        self.indiv_preds = np.array(indiv_preds).T


    def _fit_fusion_learner(self):
        self.fusion_learner.fit(self.indiv_preds, self.y[self.id == self.target])


    def _set_target(self, target):
        self.target = target


    def fit(self, X, y, id, target, indiv_num=None):
        self._load_data(X, y, id, indiv_num)
        self._set_target(target)
        self._fit_indiv_learners()
        self._predict_using_indiv_models()
        self._fit_fusion_learner()


    def predict(self, X):
        indiv_preds = []
        for k in range(self.indiv_num):
            indiv_preds.append((self.indiv_learners[k]).predict(X))
        indiv_preds = np.array(indiv_preds).T
        return self.fusion_learner.predict(indiv_preds)