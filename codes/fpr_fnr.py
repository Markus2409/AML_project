import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_fn_rate(df, column_real, column_pred, title='Miss Rate Analysis',customPalette='viridis'):
    """
    Calculates and visualizes the False Negative Rate per class for one or multiple models.

    This function iterates through each distinct class found in `column_real`. For each class, 
    it calculates the percentage of cases where the model's prediction did not match the true label
    (i.e., the miss rate). It then generates a grouped bar plot to compare these rates across 
    different models.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing both the ground truth labels and the model predictions.
    
    column_real : str
        The name of the column in `df` representing the actual/true class labels (Ground Truth).
    
    column_pred : list of str
        A list containing the names of the columns with model predictions.
        Must be a list even for a single model (e.g., ['Tree Prediction']).
        Allows comparing multiple models simultaneously (e.g., ['Tree Pred', 'SVM Pred']).
    
    title : str, default='Miss Rate Analysis'
        The title to display at the top of the plot.

    Returns
    -------
    matplotlib.pyplot
        The matplotlib.pyplot object representing the generated figure. 
        Useful if further customization (like saving the plot) is needed outside the function.
    """
    classes = sorted(df[column_real].unique()) #models and stages
    results = [] #results
    for clas in classes:
        true_stage_df = df[df[column_real] == clas] #let's separate patients of the two or more different classes
        total_cases = len(true_stage_df)    
        for pred in column_pred: #let's iterate on the prediction columns (if they are more than one for example if you are testing multiple models)
            col_name = f"{pred}"
            errors = true_stage_df[true_stage_df[col_name] != clas] #let's count how many times the model wrong
            num_errors = len(errors)
            
            # let's compute the percentage
            error_rate = (num_errors / total_cases) * 100
            
            results.append({
                'Stage': int(clas),
                'Pred': pred,
                'False Negative Rate (%)': error_rate,
                'Count': num_errors,
                'Total': total_cases
            })
    #let's plot data
    res_df = pd.DataFrame(results)
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=res_df,
        x='Stage',
        y='False Negative Rate (%)',
        hue='Pred',
        palette=customPalette,
        edgecolor='black'
    )
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3, fontsize=10, weight='bold') 
    plt.title(title, fontsize=16, weight='bold')
    plt.xlabel('True Disease Stage', fontsize=12)
    plt.ylabel('Percentage of Missed Cases (False Negatives)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')    
    plt.tight_layout()
    return plt


def plot_fp_rate(df, column_real, column_pred, title='Miss Rate Analysis', customPalette='viridis'):
    """
    Calculates and visualizes the False Positive Rate per class for one or multiple models.

    It then generates a grouped bar plot to compare these rates across 
    different models.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing both the ground truth labels and the model predictions.
    
    column_real : str
        The name of the column in `df` representing the actual/true class labels (Ground Truth).
    
    column_pred : list of str
        A list containing the names of the columns with model predictions.
        Must be a list even for a single model (e.g., ['Tree Prediction']).
        Allows comparing multiple models simultaneously (e.g., ['Tree Pred', 'SVM Pred']).
    
    title : str, default='Miss Rate Analysis'
        The title to display at the top of the plot.

    Returns
    -------
    matplotlib.pyplot
        The matplotlib.pyplot object representing the generated figure. 
        Useful if further customization (like saving the plot) is needed outside the function.
    """
    classes = sorted(df[column_real].unique()) #models and stages
    results = [] #results
    for clas in classes:
        false_stage_df = df[df[column_real] != clas] #let's take all the wrongly predicted samples
        total_cases = len(false_stage_df)    
        for pred in column_pred: #let's iterate on the prediction columns (if they are more than one for example if you are testing multiple models)
            col_name = f"{pred}"
            fp = false_stage_df[false_stage_df[col_name] == clas] #let's count how many times the model wrong
            num_fp = len(fp)
            
            # let's compute the percentage
            error_rate = (num_fp / total_cases) * 100
            
            results.append({
                'Stage': int(clas),
                'Pred': pred,
                'False Positive Rate (%)': error_rate,
                'Count': num_fp,
                'Total': total_cases
            })
    #let's plot data
    res_df = pd.DataFrame(results)
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=res_df,
        x='Stage',
        y='False Positive Rate (%)',
        hue='Pred',
        palette=customPalette,
        edgecolor='black'
    )
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3, fontsize=10, weight='bold') 
    plt.title(title, fontsize=16, weight='bold')
    plt.xlabel('True Disease Stage', fontsize=12)
    plt.ylabel('Percentage of false cases (False Positive)', fontsize=12)
    plt.ylim(0,100)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')    
    plt.tight_layout()
    return plt

