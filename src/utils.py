import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def plot_samples(x, y):
    print("\nDisplaying sample images from the dataset...")
    plt.figure(figsize=(6, 6))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(x[i], cmap='gray')
        plt.title(f"Label: {y[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.suptitle("Sample MNIST Digits", fontsize=16)
    plt.subplots_adjust(top=0.88)
    plt.show()
    print("Sample images displayed.")

def plot_predictions(model, x_test, y_test):
    print("\nGenerating model predictions and visualization...")
    predictions = model.predict(x_test, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    print(f"Predictions generated for {len(x_test)} test samples.")

    plt.figure(figsize=(12, 4))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        plt.title(f"Pred: {predicted_labels[i]}\nTrue: {y_test[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    print("Prediction visualization completed.")

    return predicted_labels

def show_confusion_matrix(y_true, y_pred):
    print("\nComputing confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix computed. Displaying visualization...")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
    print("Confusion matrix displayed.")

def print_report(y_true, y_pred):
    print("\nGenerating classification report...")
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, digits=4))
    print("Classification report completed.")