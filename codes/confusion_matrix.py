import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_cm(y_true, y_pred, model_name,  color='Blues',save_path='',labels=[1, 2, 3, 4]):
    """
    This code draws a confusion matrix
    
    Args:
    y_true: Array of the true classes
    y_pred: Array of predicted class
    model_name: Name of the model that you used (to insert it into the title)
    color: is used to choose the palette of the confusion matrix (blue is default)
    save_path: is used to indicate the path where to save the cm... if left empty the figure is not saved.
    labels: list of etiquettes (it uses the classes indicated in y_true per default)

    """

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=color, 
                xticklabels=labels, yticklabels=labels,
                cbar=False, annot_kws={"size": 14}) # Grandezza font numeri
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16, pad=20)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    if save_path!='':
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
