from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

class Evaluation:
    def mse(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

    def rmse(self, y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def r2(self, y_true, y_pred):
        return r2_score(y_true, y_pred)
