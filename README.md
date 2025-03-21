# 🚀 Dataset Not Included

## 📥 Download the Dataset
Due to GitHub’s file size limits, the dataset is **not included** in this repository.

🔗 **Download it here:** [Google Drive / OneDrive Link](https://drive.google.com/file/d/1tAOXF57zB00HMooda06D2UNXjNqTxO12/view?usp=sharing)

## 📂 How to Use
1. Download `00_dataset.zip` from the link above.
2. Extract it inside the root directory of this repository.

## Folder Structure
- `01_content` contains the notebooks related to textual analysis of the review content.

- `02_contextual` contains the notebooks related to contextual analysis of the review metadata.
    - `a1_restaurant_feature_engineering.ipynb` and `a2_user_feature_engineering.ipynb` contains the preliminary feature engineering experiments and manual analysis.
    - `RestaurantFeatureEngineering.py`, `UserFeatureEngineering.py` and `FeatureEngineer.py` are the final feature engineering files used in the model experiments.
    - `b1_contextualmodel_preliminary.ipynb` contains the initial model evaluation, comparing the performance with and without feature engineering.
    - `b2_contextualmodel_kfold.ipynb` consists of the streamlined version of model evaluation using `Pipeline`, along with the K-Fold Cross Validation evaluation.
    - `b3_feature_selection.ipynb` consists of the feature selection and reduction methods used on the set of engineered features.
    - `b3_modelling_kfold_feature_engineering.ipynb` contains the non-streamlined version of the K-Fold Cross Validation model evaluation for the multiple feature selection methods.
    - `b4_modelling_kfold_full_pipeline.ipynb` consists of the evaluation of every combination of feature selection, scaler and model.
    - `b5_final_model_evaluation.ipynb` contains the model selected by the selection formula and the evaluation of said model.

- `03_combined` contains the notebooks and relevant results for the decision-level fusion of final chosen models from `01` and `02`.
