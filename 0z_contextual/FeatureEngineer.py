from UserFeatureEngineering import UserEngineer
from RestaurantFeatureEngineering import RestaurantEngineer


class CombinedEngineer():
    def __init__(self):
        self.user_engineer = None
        self.res_engineer = None

    
    def fit(self, train_data, y=None):
        self.user_engineer = UserEngineer(train_data)
        self.user_features = self.user_engineer.aggregateTrainFeatures()

        self.res_engineer = RestaurantEngineer(train_data)
        self.res_features = self.res_engineer.aggregateTrainFeatures()
        return self


    def transform(self, X, y=None):
        if self.res_engineer is None or self.user_engineer is None:
            raise ValueError("Call fit() before transform()")
        
        res_X = self.res_engineer.updateTestRestaurants(X)
        new_X = self.user_engineer.updateTestUsers(res_X)
        new_X = new_X.drop(columns=["user_id", "prod_id"], errors="ignore")
        return new_X
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)