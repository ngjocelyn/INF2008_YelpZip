import pandas as pd

class RestaurantEngineer():
    def __init__(self, train_data):
        self.train_set = train_data.copy()
    
    def aggregateTrainFeatures(self):
        df = self.train_set

        # Convert date column to datetime for calculations
        df["date"] = pd.to_datetime(df["date"])

        # Compute restaurant-based features
        restaurant_review_counts = df.groupby("prod_id")["rating"].count().rename("total_reviews_for_restaurant")
        restaurant_avg_rating = df.groupby("prod_id")["rating"].mean().rename("avg_rating_for_restaurant")
        restaurant_rating_std = df.groupby("prod_id")["rating"].std().rename("std_dev_rating_for_restaurant")
        restaurant_median_rating = df.groupby("prod_id")["rating"].median().rename("median_rating_for_restaurant")

        # Compute review frequency per restaurant
        restaurant_review_dates = df.groupby("prod_id")["date"].agg(["min", "max", "count"])

        # Review frequency is calculated as (latest_review_date - earliest_review_date) / total_reviews
        '''High Values (e.g., 30+ days per review) → LOW Activity
        Means the restaurant gets infrequent reviews.
        This is expected for small/local restaurants.
        Not necessarily suspicious unless combined with high rating standard deviation.
        2. Moderate Values (e.g., 3-15 days per review) → NORMAL Activity
        Restaurants typically get a review every few days to a week.
        Popular places should fall in this range.
        3. Low Values (e.g., <1 day per review) → HIGH Activity
        Means the restaurant is getting multiple reviews per day.
        This could be organic (high foot traffic places like chains) or suspicious (fake reviews).
        Suspicious if:
        There is a sudden burst of reviews after inactivity.
        A large percentage of reviews come from new users.
        Many reviews have similar timestamps or wording.'''

        restaurant_review_dates["review_frequency_for_restaurant"] = (restaurant_review_dates["max"] - restaurant_review_dates["min"]).dt.days / restaurant_review_dates["count"].clip(lower=2)
        restaurant_review_dates = restaurant_review_dates["review_frequency_for_restaurant"]

        # Compute unique and repeat reviewers count
        '''All reviewers are unique, no reviewer reviewed the same restaurant twice'''

        # Compute Extreme Rating Index
        '''0 → All reviews are 3-star (perfectly neutral).
        1 → Equal mix of 2-star, 3-star, and 4-star reviews.
        2 → All reviews are either 1-star or 5-star (highly polarized).'''

        df["rating_deviation"] = abs(df["rating"] - 3)  # Distance from neutral (3-star)
        extreme_rating_index = df.groupby("prod_id")["rating_deviation"].mean().rename("extreme_rating_index")


        # Merge computed features
        restaurant_features = df[["prod_id"]].drop_duplicates()
        restaurant_features = restaurant_features.merge(restaurant_review_counts, on="prod_id", how="left")
        restaurant_features = restaurant_features.merge(restaurant_avg_rating, on="prod_id", how="left")
        restaurant_features = restaurant_features.merge(restaurant_rating_std, on="prod_id", how="left")
        restaurant_features = restaurant_features.merge(restaurant_median_rating, on="prod_id", how="left")
        restaurant_features = restaurant_features.merge(restaurant_review_dates, on="prod_id", how="left")
        restaurant_features = restaurant_features.merge(extreme_rating_index, on="prod_id", how="left")

        # Fill NaN values for standard deviation (caused by single reviews) with 0
        restaurant_features["std_dev_rating_for_restaurant"] = restaurant_features["std_dev_rating_for_restaurant"].fillna(0)

        self.train_features = pd.DataFrame(restaurant_features)
        return self.train_features

    # Function for a new user
    def updateTestRestaurants(self, test_X):
        # Merge on restaurant data, existing restaurant info from train set will be merged
        new_X = test_X.merge(self.train_features, on="prod_id", how="left")

        # Define default values if there are restaurants that do not exist in the train dataset
        new_X.fillna({
            "total_reviews_for_restaurant": 0,
            "avg_rating_for_restaurant": self.train_features["avg_rating_for_restaurant"].median(),
            "std_dev_rating_for_restaurant": self.train_features["std_dev_rating_for_restaurant"].median(),
            "median_rating_for_restaurant": self.train_features["median_rating_for_restaurant"].median(),
            "review_frequency_for_restaurant": 0,
            "extreme_rating_index": self.train_features["extreme_rating_index"].median(),
            # "rating_deviation": train_features["rating_deviation"].median()
        }, inplace=True)
        
        int_cols = ["total_reviews_for_restaurant"]
        new_X[int_cols] = new_X[int_cols].astype(int)
        return new_X