import pandas as pd

class UserEngineer():
    def __init__(self, train_data):
        self.train_set = train_data.copy()
    
    def aggregateTrainFeatures(self):
        df = self.train_set
        
        # Ensure 'date' column is in datetime format
        df['date'] = pd.to_datetime(df['date'])

        # Aggregate user features
        user_features = df.groupby('user_id').agg({
            'prod_id': 'count',  # no. of restaurant reviews per user
            'rating': ['mean', 'min', 'max', 'std'],  # rating statistics
            'date': ['min', 'max']  # First and last review dates
        })

        # std will be NA for people with only 1 review, so fill it with reviews.
        user_features[('rating', 'std')] = user_features[('rating', 'std')].fillna(0)

        # unique days active (no. of days the user has made a rating/review)
        user_activity = df.groupby('user_id')['date'].nunique()

        user_features[('unique_days_active', '')] = user_activity

        # Calculate review timespan
        user_features['review_timespan'] = (user_features[('date', 'max')] - user_features[('date', 'min')]).dt.days

        # Avoid division by zero for users with only one review
        user_features['review_timespan'] = user_features['review_timespan'].replace(0, 1)

        # Compute average reviews per day
        user_features['avg_reviews_per_day'] = user_features[('prod_id', 'count')] / user_features['review_timespan']

        # Compute percentage of active days against user existence date
        user_features['user_active_percentage'] = user_features['unique_days_active'] / user_features['review_timespan']

        # round floats to 3dp
        user_features = user_features.round(3)

        # flatten df
        user_features.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in user_features.columns]
        user_features.fillna(0)

        # Rename columns
        user_features.rename(columns={
            'prod_id_count': 'user_restaurants_reviewed', 
            'date_min': 'user_earliest', 
            'date_max': 'user_latest',
            'review_timespan': 'user_review_timespan',
            'unique_days_active': 'user_days_active',
            'avg_reviews_per_day': 'users_avg_per_day'
        }, inplace=True)

        user_features['user_earliest'] = user_features['user_earliest'].astype('int64') // 10**9
        user_features['user_latest'] = user_features['user_latest'].astype('int64') // 10**9

        self.train_features = user_features.reset_index()
        return self.train_features
    
    # Function for a new user
    def updateTestUsers(self, test_X):
        # Merge on User ID data, if any users already exist in the trainset
        new_X = test_X.merge(self.train_features, on="user_id", how="left")

        # Ensure 'date' is in datetime format
        new_X['date'] = pd.to_datetime(new_X['date'], errors='coerce')

        # Convert to UNIX timestamp correctly
        new_X['date'] = new_X['date'].apply(lambda x: int(x.timestamp()) if pd.notnull(x) else 0)

        # Define default values if there are users that do not exist in the train dataset
        new_X.fillna({
            "rating_mean": 3.0,
            "rating_min": 3.0,
            "rating_max": 3.0,
            "rating_std": 0.0,
            "user_earliest": new_X['date'],
            "user_latest": new_X['date'],
            "user_days_active": 0,
            "user_review_timespan": 1,
            "users_avg_per_day": 0.0,
            "user_active_percentage": 0.0,
            "user_restaurants_reviewed": 0,
        }, inplace=True)
        int_cols = ["user_earliest", "user_latest", "user_restaurants_reviewed", "user_review_timespan", "user_days_active"]
        new_X[int_cols] = new_X[int_cols].astype(int)
        return new_X