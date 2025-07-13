import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, test_gen):
    """
    Evaluate model performance and generate metrics
    """
    y_true = test_gen.labels
    y_pred = np.argmax(model.predict(test_gen), axis=1)
    class_names = list(test_gen.class_indices.keys())
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    return y_true, y_pred

def average_fusion(models, test_gen):
    """
    Perform average fusion of multiple models' predictions
    """
    # Get all predictions
    predictions = [model.predict(test_gen) for model in models]
    
    # Average predictions
    avg_preds = np.mean(predictions, axis=0)
    y_pred = np.argmax(avg_preds, axis=1)
    y_true = test_gen.labels
    
    # Calculate accuracy
    accuracy = np.mean(y_pred == y_true) * 100
    print(f"Fused Model Accuracy: {accuracy:.2f}%")
    
    return y_pred