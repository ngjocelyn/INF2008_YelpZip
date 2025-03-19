# 🚀 Dataset Not Included

## 📥 Download the Dataset
Due to GitHub’s file size limits, the dataset is **not included** in this repository.

🔗 **Download it here:** [Google Drive / OneDrive Link](https://drive.google.com/file/d/1tAOXF57zB00HMooda06D2UNXjNqTxO12/view?usp=sharing)

## 📂 How to Use
1. Download `00_dataset.zip` from the link above.
2. Extract it inside the root directory of this repository.

## A few things to take note:
1. Try out on Naive Bayes (MultinomialNB and LogisticRegression first).
2. Do MinMaxScaling
3. Sampling (SMOTE or SMOTE Tomek) might need to be done (can go and google/gpt to see how it can be done)

## Folder Structure
`01_content` contains the notebooks related to textual analysis of the review content.

`02_contextual` contains the notebooks related to contextual analysis of the review metadata.
- `a1_restaurant_feature_engineering.ipynb` and `a2_user_feature_engineering.ipynb` contains the preliminary feature engineering experiments and manual analysis.
- `RestaurantFeatureEngineering.py`, `UserFeatureEngineering.py` and `FeatureEngineer.py` are the final feature engineering files used in the model experiments.
- `b1_contextualmodel_preliminary.ipynb` contains the initial model evaluation, comparing the performance with and without feature engineering.
- `b2_contextualmodel_kfold.ipynb` consists of the streamlined version of model evaluation using `Pipeline`, along with the K-Fold Cross validation evaluation.


`03_combined` contains the notebooks and relevant results for the decision-level fusion of final chosen models from `01` and `02`.