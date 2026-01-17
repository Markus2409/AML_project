import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
def visualize_feature_importance(model, feature_names, model_type, X_val=None, y_val=None, title=None, export_path=None):
    """
    Generates and plots feature importance visualizations based on the specific model type.
    
    This function handles three specific scenarios:
    1. 'tree': Extracts Gini importance from Random Forest models.
    2. 'lr': Extracts coefficients specifically for Class 3 (Stage 4 - Cirrhosis) from Logistic Regression.
    3. 'svm' (or 'permutation'): Calculates Permutation Importance (agnostic method), required for Black-Box models like SVM RBF.

    Parameters
    ----------
    model : sklearn estimator or pipeline
        The trained model object (must contain steps named 'rf' for trees or 'lr' for logistic regression).
    feature_names : list or array-like
        List of strings containing the names of the features.
    model_type : str
        The type of analysis to perform. Options: 'tree', 'lr', 'svm'.
    X_val : array-like, optional
        Validation/Test data (features). Required ONLY if model_type is 'svm'.
    y_val : array-like, optional
        Validation/Test labels (target). Required ONLY if model_type is 'svm'.
    title : str, optional
        Custom title for the plot. If None, a default title is generated.
    export_path : str, optional
        If provided, saves the figure to the specified path (e.g., '../figures/plot.png').

    Returns
    -------
    None
        Displays the plot and optionally saves it.
    """
    
    plt.figure(figsize=(10, 6))
    
    # --- CASE 1: RANDOM FOREST (Gini Importance) ---
    if model_type == "tree":
        if title is None: title = "Random Forest: Gini Importance"
        
        # Extract importance from the 'rf' step of the pipeline
        importances = model.named_steps['rf'].feature_importances_
        
        # Create DataFrame for sorting
        df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        df_imp = df_imp.sort_values(by='Importance', ascending=False)
        
        # Plot
        sns.barplot(
            x='Importance', 
            y='Feature', 
            data=df_imp, 
            hue='Feature', 
            palette='Oranges_r', 
            legend=False
        )
        plt.xlabel("Gini Importance (Weight)")
        plt.grid(axis='x', linestyle='--', alpha=0.5)

    # --- CASE 2: LOGISTIC REGRESSION (Heatmap for ALL Stages) ---
    elif model_type == "lr":
        if title is None: title = "Logistic Regression: Coefficients across ALL Stages"
        
        # Extract ALL coefficients
        coeffs = model.named_steps['lr'].coef_
        
        # Create DataFrame
        df_coeffs = pd.DataFrame(
            coeffs, 
            columns=feature_names, 
            index=["Stage 1", "Stage 2", "Stage 3", "Stage 4"]
        )
        
        # Transpose to have Features on rows
        df_coeffs = df_coeffs.T
        
        # Sort by Stage 4
        df_coeffs = df_coeffs.sort_values(by="Stage 4", ascending=False)
        
        # Plot Heatmap con Label sulla colorbar
        sns.heatmap(
            df_coeffs, 
            annot=True,     
            fmt=".2f",      
            cmap="coolwarm", 
            center=0,       
            linewidths=0.5,
            cbar_kws={'label': 'Coefficient Value\n(Red = Positive Association/Risk, Blue = Negative Association/Protection)'} # <--- NUOVA LABEL
        )
        plt.ylabel("Biomarker")
        plt.xlabel("Disease Stage")

    # --- CASE 3: SVM / PERMUTATION IMPORTANCE ---
    elif model_type == "svm" or model_type == "permutation":
        if title is None: title = "SVM: Permutation Feature Importance"
        
        if X_val is None or y_val is None:
            raise ValueError("For 'svm' or 'permutation' type, you MUST provide X_val and y_val.")

        # Calculate permutation importance (repeating 10 times for statistical robustness)
        result = permutation_importance(
            model, X_val, y_val, 
            n_repeats=10, 
            random_state=42, 
            n_jobs=-1, 
            scoring='matthews_corrcoef'
        )
        
        # Sort indices by mean importance
        sorted_idx = result.importances_mean.argsort()
        
        # Create Boxplot
        plt.boxplot(
            result.importances[sorted_idx].T,
            vert=False,
            labels=np.array(feature_names)[sorted_idx]
        )
        plt.xlabel("Decrease in MCC Score")
        plt.grid(axis='x', linestyle='--', alpha=0.5)

    else:
        print(f"Error: model_type '{model_type}' not recognized. Use 'tree', 'lr', or 'svm'.")
        return

    # Final layout adjustments
    plt.title(title, fontsize=15, pad=15)
    plt.tight_layout()
    
    # Save if path is provided
    if export_path:
        plt.savefig(export_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {export_path}")
        
    plt.show()


def plot_multiclass_roc_curve(models_dict, X_test_dict, y_test, target_class=4, export_path=None):
    """
    Plots ROC curves for multiple models on the same figure for a specific target class vs Rest.
    
    Parameters
    ----------
    models_dict : dict
        Dictionary where Key = Model Name (str) and Value = Model Object.
        Example: {'Logistic Regression': loaded_lr, 'SVM': loaded_svm}
    X_test_dict : dict
        Dictionary where Key = Model Name and Value = X_test data specific for that model.
        Example: {'Logistic Regression': x_bench_lr, 'SVM': x_bench_svm}
    y_test : array-like
        True labels for the test set.
    target_class : int, default=4
        The class to treat as "Positive" (e.g., 4 for Cirrhosis).
    export_path : str, optional
        Path to save the figure.
    """
    
    plt.figure(figsize=(9, 7))
    
    # Binarize labels (One-vs-Rest)
    y_test_bin = label_binarize(y_test, classes=[1, 2, 3, 4])
    class_index = target_class - 1
    
    for model_name, model in models_dict.items():
        X_test = X_test_dict[model_name]
        
        # Get scores (Probability or Decision Function)
        try:
            y_score = model.predict_proba(X_test)
        except AttributeError:
            y_score = model.decision_function(X_test)
            
        # Compute ROC
        fpr, tpr, _ = roc_curve(y_test_bin[:, class_index], y_score[:, class_index])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')
        
    # Random guess line
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    
    # Aesthetics
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title(f'ROC Curves - Ability to Detect Stage {target_class}', fontsize=15)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)
    
    if export_path:
        plt.savefig(export_path, dpi=300, bbox_inches='tight')
        print(f"ROC Plot saved to {export_path}")
        
    plt.show()


