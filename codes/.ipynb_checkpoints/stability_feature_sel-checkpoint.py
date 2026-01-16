import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import matthews_corrcoef
from imblearn.ensemble import BalancedRandomForestClassifier
import numpy as np
def performance_on_subset(subset_features, x_train, y_train, x_val, y_val, pipeline):
    """
    Evaluates the performance of a given model restricted to a specific subset of features.

    Parameters:
    - subset_features (list): List of column names to be used for training/validation.
    - X_train (pd.DataFrame): Training feature matrix.
    - y_train (pd.Series): Training target vector.
    - X_val (pd.DataFrame): Validation feature matrix.
    - y_val (pd.Series): Validation target vector.
    - model (estimator): A scikit-learn compatible estimator (e.g., SVC, RandomForest,LogisticRegression) to be trained and evaluated.

    Returns:
    - mcc (float): The Matthews Correlation Coefficient score on the validation set.
    """
    #find the col number of a specific feature  
    #take the column of the corrispective feature
    Xtr = x_train[subset_features] 
    Xva = x_val[subset_features]
    pipeline.fit(Xtr, y_train)     # train the svm on training data
    y_pred = pipeline.predict(Xva) # predict on validation data
    mcc = matthews_corrcoef(y_val, y_pred) # compute MCC 
    return mcc  # mcc on VALIDATION

    
def feat_sel(X_training,y_training,x_validation,y_validation, estimator, rf):
    """
    Performs a feature selection procedure using a Random Forest (that can be chosen by the user; e.g. Balanced or Classic) for ranking 
    and an estimator (e.g., SVM ,RF, LogisticRegression...) for validation performance.

    The function ranks features by Gini Importance (using the rf model) and then 
    iteratively evaluates the estimator's performance on the Top-K features 
    to find the optimal number of features (Best K).

    Parameters:
    - X_train (pd.DataFrame): Training feature matrix.
    - y_train (pd.Series): Training target vector.
    - X_val (pd.DataFrame): Validation feature matrix.
    - y_val (pd.Series): Validation target vector.
    - estimator (object): The model used to validate the subset performance (e.g., SVC).
    - rf_model (object): The Tree-based model used to calculate feature importance (e.g., RandomForest).

    Returns:
    - gini_df (pd.DataFrame): Dataframe containing features sorted by importance.
    - best_k (int): The number of features that maximized the MCC score.
    - ks (list): List of all k values tested (x-axis for plotting).
    - curve (list): List of MCC scores corresponding to each k (y-axis for plotting).
    """
    feature_name=X_training.columns.tolist()
    rf.fit(X_training, y_training)  # fit on train data
    
    # create a Series with feature name and the corresponding feature importance based on Gini impurity
    gini_imp = pd.Series(rf.feature_importances_, index=feature_name).sort_values(ascending=False) 
    gini_df = gini_imp.reset_index()
    
    # name the columns
    gini_df.columns = ["feature", "importance"
                      ]
        # FIND THE BEST NUMBER OF FEATURES (K)
    # create a list with all the features
    ks = list(range(2, X_training.shape[1]+1))
    curve = []

    #iterate over possible k-features to evaluate how the performance goes on
    for k in ks:
        subset = gini_df["feature"].head(k).tolist()
        mcc_k = performance_on_subset(subset, X_training, y_training, x_validation, y_validation, estimator)
        curve.append(mcc_k)
        
    # find the k that maximizes the MCC 
    best_k_idx = int(np.argmax(curve))
    best_k = ks[best_k_idx]

    return gini_df, best_k, ks, curve