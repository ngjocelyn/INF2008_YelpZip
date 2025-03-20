from UserFeatureEngineering import UserEngineer
from RestaurantFeatureEngineering import RestaurantEngineer


class CombinedEngineer():
    def __init__(self, drop_columns=None):
        
        
        self.user_engineer = None
        self.res_engineer = None
        # :param drop_columns: List of column names to drop from the dataset.
        self.drop_columns = drop_columns if drop_columns else []
        self.final_columns = None  # Stores the final feature names after fit()

    
    def fit(self, train_data, y=None):
        self.user_engineer = UserEngineer(train_data)
        self.user_features = self.user_engineer.aggregateTrainFeatures()

        self.res_engineer = RestaurantEngineer(train_data)
        self.res_features = self.res_engineer.aggregateTrainFeatures()
        
        # Merge and determine final feature set
        temp_X = self.user_engineer.updateTestUsers(self.res_engineer.updateTestRestaurants(train_data))
        
         # Define final columns after dropping
        default_drop_columns = ["user_id", "prod_id", "review"]
        columns_to_drop = list(set(default_drop_columns + self.drop_columns))
        
        self.final_columns = [col for col in temp_X.columns if col not in columns_to_drop]
        
        return self


    def transform(self, X, y=None):
        if self.res_engineer is None or self.user_engineer is None:
            raise ValueError("Call fit() before transform()")
        
        res_X = self.res_engineer.updateTestRestaurants(X)
        new_X = self.user_engineer.updateTestUsers(res_X)
        
        # Drop specified columns
        new_X = new_X[self.final_columns]  # Ensure feature consistency
        
        return new_X
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)