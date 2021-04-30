class BayesianClassifier(BaseEstimator, ClassifierMixin):  
    """Classify samples based on the distance from the mean feature value"""

    def __init__(self):
        self.X_mean_ = None
        self.X_var_ = None
        self.P_priors = None
        
    def fit(self, X, y, mono_var=False):
         
        D = 10*[None]
        P = 10*[0]
        N = len(y)
        N_ = 10*[0]
        self.X_mean_ = 256*[0]
        self.X_var_ = 256*[0]
        self.P_priors = np.zeros(10)        
        
        for i in range(10):
            (indices,) = np.where(y == i)
            D[i] = X[indices]
            N_[i] = len(indices)
            self.X_mean_[i] = np.mean(D[i], axis=0)
            self.X_var_[i] = np.var(D[i], axis=0)
            self.P_priors[i] = N_[i]/N

        if mono_var:
            self.X_var_ = 10*[np.full((256,), 1)]
        
        return self


    def predict(self, X):

        def truncated_gaussian_naive_log_likelihood(Mu, Var, X_n): # The arguments are all vectors
            (indices,) = np.where(Var != 0) # We do away with zero variance
            Mu_, X_n_ = Mu[indices], X_n[indices]
            Var_ = Var[indices]
            return  -np.sum((X_n_-Mu_)** 2 / Var_) - np.sum( np.log(Var_) )/2 
        
        y_pred = np.zeros(X.shape[0])
        for n in range(X.shape[0]):
            Bayes_nominators = [np.log(self.P_priors[i]) + truncated_gaussian_naive_log_likelihood(self.X_mean_[i],self.X_var_[i],X[n]) for i in range(10)]    
            y_pred[n] = np.argmax(Bayes_nominators)
         
        return y_pred
        
    def score(self, X, y):
        y_pred = clf.predict(X)
        num_of_correct_predictions = np.sum(y_pred == y)
              

        return num_of_correct_predictions/len(y)