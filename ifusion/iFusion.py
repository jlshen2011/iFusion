
# coding: utf-8

# In[49]:


def performance(mean_all, cov_all, truth, alpha = 0.05, rounding = 5, dist = "Gaussian"):
    p = len(theta)
    mse, coverage, width = [], [], []
    multiplier = norm.ppdf(1 - alpha)
    for j in range(p):
        mse.append(np.mean(np.square(mean_all[:, j] - truth[j])))
        ci_up = mean_all[:, j] + multiplier * np.sqrt(cov_all[:, j, j])
        ci_low = mean_all[:, j] - multiplier * np.sqrt(cov_all[:, j, j])
        coverage.append(np.mean(ci_up >= truth[j] and ci_low <= truth[j]))
        width.append(np.mean(multiplier * cov_all[:, j, j]))
    return mse, coverage, width


# In[614]:


import numpy as np
from numpy import random
from scipy.stats import norm
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

class iFusion:
    """Individualized fusion learning for linear models"""
    
    def __init__(self, bandwidth = None, bandwidth_path = np.arange(0.01, 5, 0.01), cv_num = 5, early_stopping_rounds = None, epsilon = 0.1, kernel = "uniform", loss_func = "mse", screen_prop = None, tau = -1/4, weight_vector = None):       
        self.params = {
            "cv_num": cv_num,
            "early_stopping_rounds": early_stopping_rounds if early_stopping_rounds is not None else len(bandwidth_path),
            "epsilon": epsilon,
            "loss_func": loss_func,
            "screen_prop": screen_prop,
            "kernel": kernel,
            "tau": tau,
            "bandwidth_path": bandwidth_path
        }            
        self.bandwidth = bandwidth
        self.weight_vector = weight_vector

        
    def loadData(self, X, y):
        """load data
        
        Parameters
        ----------
        X : list of numpy array
        y : list of numpy array
        """
        self.X = X
        self.y = y
        self.n = np.array([len(x) for x in X])
        self.n_total = sum(self.n)
        self.p = X[0].shape[1]
        self.K = len(X)

        
    def setTarget(self, target_id):
        """set target"""
        self.target_id = target_id
        
                
    def indivCD(self, X, y):
        """compute individual CD
        
        Parameters
        ----------
        X : numpy array
        y : numpy array
        """
        n, p = X.shape
        M = np.linalg.inv(X.T.dot(X))
        indiv_mean = M.dot(X.T).dot(y)
        indiv_cov = np.sum(np.square(y - X.dot(indiv_mean))) / (n - p) * M
        return indiv_mean, indiv_cov

    
    def allIndivCD(self):
        """compute individual CD for each individual"""
        self.indiv_mean_all = np.zeros((self.K, self.p))
        self.indiv_cov_all = np.array([np.identity(self.p) for _ in range(self.K)])
        for k in range(self.K):
            self.indiv_mean_all[k, :], self.indiv_cov_all[k, :, :] = self.indivCD(self.X[k], self.y[k])    


    def weightVector(self, indiv_mean_all, indiv_cov_all, n, bandwidth):
        """compute screen weight vector given bandwidth parameter(s)"""
        bandwidth = np.array([bandwidth]).flatten()
            
        dist = float("inf") * np.ones((len(bandwidth), self.K))
        keep = range(self.K)
        n_hm = 2 * n[self.target_id] * n / (n[self.target_id] + n)
        
        if self.params["screen_prop"] is not None:
            dist_est = np.apply_along_axis(lambda x: np.sum((x - indiv_mean_all[self.target_id, :]) ** 2), 1, indiv_mean_all)
            keep, = np.where(dist_est <= np.percentile(dist_est, self.params["screen_prop"] * 100, interpolation = "higher"))
        for k in keep:
            tmp = np.sqrt((indiv_mean_all[self.target_id, :] - indiv_mean_all[k, :]).T.dot(np.linalg.inv(indiv_cov_all[self.target_id, :, :] + indiv_cov_all[k, :, :])).dot(indiv_mean_all[self.target_id, :] - indiv_mean_all[k, :])) / (n_hm[k] * self.p) / n_hm[k] ** self.params["tau"]
            dist[:, k] = tmp / bandwidth
                        
        if self.params["kernel"] == "uniform":
            weight_vector = 1 * (np.abs(dist) <= 1)
        elif self.params["kernel"] == "triangular":
            weight_vector = (1 - dist ** 2) * np.where(np.abs(dist) <= 1, 1, 0)
        elif self.params["kernel"] == "gaussian":
            weight_vector = np.exp(-dist ** 2 / 2)
        
        if len(bandwidth) == 1:
            return weight_vector[0]
        return weight_vector

    
    def combineCD(self, indiv_mean_all, indiv_cov_all, weight_vector):
        """combine individual CDs into a combined CD given the weight vector"""
        tmp1, tmp2, tmp3 = np.zeros((self.p, self.p)), np.zeros(self.p), np.zeros((self.p, self.p))
        for k, w in enumerate(weight_vector):
            if w != 0:
                tmp0 = np.linalg.inv(indiv_cov_all[k, :, :])
                tmp1 += w * tmp0
                tmp2 += w * tmp0.dot(indiv_mean_all[k, :])
                tmp3 += w ** 2 * tmp0
        tmp4 = np.linalg.inv(tmp1)
        comb_mean, comb_cov = tmp4.dot(tmp2), tmp4.dot(tmp3).dot(tmp4)
        return comb_mean, comb_cov

    
    def tuneBandwidth(self):
        """tune bandwidth parameter based on cross-validation""" 
        bandwidth_path_len = len(self.params["bandwidth_path"])
        loss = float("inf") * np.ones((bandwidth_path_len, self.params["cv_num"]))
        loss_mean = []
        loss_std = []
        
        kf = KFold(n_splits = self.params["cv_num"])  
        kf_splits = [kf.split(self.X[k], self.y[k]) for k in range(self.K)]
        
        indiv_mean_all_cv = np.zeros((bandwidth_path_len, self.K, self.p))
        indiv_cov_all_cv = np.zeros((bandwidth_path_len, self.K, self.p, self.p))
        X_target_test, y_target_test = [], []         
                                 
        for k, kf_split in enumerate(kf_splits):
            fold = 0                        
            for train_index, test_index in kf_split:
                indiv_mean_all_cv[fold, k, :], indiv_cov_all_cv[fold, k, :, :] = self.indivCD(self.X[k][train_index, :], self.y[k][train_index])     
                fold += 1
                if k == self.target_id:
                    X_target_test.append(self.X[k][test_index, :])
                    y_target_test.append(self.y[k][test_index])
                                
        weight_vector_all = np.zeros((self.params["cv_num"], bandwidth_path_len, self.K))
        for fold in range(self.params["cv_num"]):
            weight_vector_all[fold, :, :] = self.weightVector(indiv_mean_all_cv[fold, :, :], indiv_cov_all_cv[fold, :, :, :], self.n * (self.params["cv_num"] - 1) / self.params["cv_num"], self.params["bandwidth_path"])
                                         
        for i in range(bandwidth_path_len):
            for fold in range(self.params["cv_num"]):
                comb_mean_new, _ = self.combineCD(indiv_mean_all_cv[fold, :, :], indiv_cov_all_cv[fold, :, :, :], weight_vector_all[fold, i, :])
                if self.params["loss_func"] == "mse":
                    loss[i, fold] = np.mean((y_target_test[fold] - X_target_test[fold].dot(comb_mean_new)) ** 2)
            loss_mean.append(np.mean(loss[i, :]))
            loss_std.append(np.std(loss[i, :]) / np.sqrt(self.params["cv_num"]))
            if i % 10 == 0:
                print("bandwidth: {}, loss_mean: {}, loss_std: {}".format(round(self.params["bandwidth_path"][i], 2),  round(loss_mean[i], 2), round(loss_std[i], 2)))
            if i == 0:
                count = 0
                loss_min = loss_max = loss_mean[0]
            loss_max = max(loss_max, loss_mean[i])
            if loss_min >= loss_mean[i]:
                loss_min = loss_mean[i]
                count = 0
            buffer = self.params["epsilon"] * loss_min
            if loss_mean[i] > loss_min + buffer:
                count += 1
            if count >= self.params["early_stopping_rounds"]:
                break
        
        i_min, = np.where(loss_mean == loss_min)
        i_min = int(np.floor(np.median(i_min)))
        i_opt = np.where(loss_mean <= loss_mean[i_min] + loss_std[i_min])
        i_opt = int(np.floor(np.median(i_opt)))
        return loss_mean, loss_std, self.params["bandwidth_path"][i_opt]                          
    
                           
    def fit(self, X, y, target_id):
        """main function"""                                  
        self.loadData(X, y)                          
        self.allIndivCD()
        self.setTarget(target_id)
                          
        if self.weight_vector is None:
            if self.bandwidth is None:
                self.loss_mean, self.loss_std, self.bandwidth = self.tuneBandwidth()
            self.weight_vector = self.weightVector(self.indiv_mean_all, self.indiv_cov_all, self.n, self.bandwidth)

        self.comb_mean, self.comb_cov = self.combineCD(self.indiv_mean_all, self.indiv_cov_all, self.weight_vector)
                           
                           
    def getIndivCD(self):
        """get individual CD for the target individual"""
        return self.indiv_mean_all[self.target_id, :], self.indiv_cov_all[self.target_id, :, :]

                           
    def getCombCD(self):
        """get combined CD for the target individual"""
        return self.comb_mean, self.comb_cov  
    
    def getWeightVector(self):
        return self.weight_vector
    
    def plotTuning(self):
        n = len(self.loss_mean)
        plt.figure(1, figsize = (6,4) )
        plt.errorbar(self.params["bandwidth_path"][:n], self.loss_mean, fmt='ro', label="data",xerr=0, yerr=self.loss_std, ecolor='black')
        plt.xlabel('x')
        plt.ylabel('transverse displacement')


# In[615]:


X = [np.array(np.random.randn(1000)).reshape(500, 2) for _ in range(20)]

y = []
for k in range(10):
    y.append(np.dot(X[k], np.array([1,2])) + random.randn(500))
for k in range(10, 20):
    y.append(np.dot(X[k], np.array([5,1])) + random.randn(500))


# In[616]:


m = iFusion(early_stopping_rounds = 2, cv_num = 10)


# In[617]:


m.fit(X, y, 1)


# In[618]:


m.getIndivCD()


# In[619]:


m.getCombCD()


# In[612]:


m.getWeightVector()


# In[613]:


m.plotTuning()

