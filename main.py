from src.data_loader import load_data, preprocess_data
from src.model_builder import build_model
from src.train import train_model, evaluate_model
from src.utils import plot_samples, plot_predictions, show_confusion_matrix, print_report
from keras.models import save_model

if __name__ == "__main__":
    print("Starting MNIST classification pipeline...")
    print("\n[1/5] Loading dataset...")
    (x_train, y_train), (x_test, y_test) = load_data()
    
    """print("[2/6] Displaying sample images...")
    plot_samples(x_train, y_train)"""

    print("[2/5] Preprocessing data...")
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)

    print("[3/5] Building model architecture...")
    model = build_model()
    print("Model Summary:")
    model.summary()

    print("[4/5] Training model...")
    history = train_model(model, x_train, y_train)
    
    print("[5/5] Evaluating model...")
    acc = evaluate_model(model, x_test, y_test)
    print(f"Model evaluation complete. Test accuracy: {acc:.4f}")

    """print("[7/7] Generating predictions and metrics...")
    predicted_labels = plot_predictions(model, x_test, y_test)
    show_confusion_matrix(y_test, predicted_labels)
    print_report(y_test, predicted_labels)"""

    save_model(model, "models/mnist_cnn_model.h5")
    print("\nPipeline completed successfully!")
    print("Model saved as models/mnist_cnn_model.h5")