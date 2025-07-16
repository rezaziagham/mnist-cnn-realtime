from src.data_loader import load_data, preprocess_data
from src.model_builder import build_model
from src.train import train_model, evaluate_model
from src.utils import plot_samples, plot_predictions, show_confusion_matrix, print_report
from keras.models import save_model

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_data()
    plot_samples(x_train, y_train)

    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)

    model = build_model()
    model.summary()

    history = train_model(model, x_train, y_train)
    acc = evaluate_model(model, x_test, y_test)

    predicted_labels = plot_predictions(model, x_test, y_test)
    show_confusion_matrix(y_test, predicted_labels)
    print_report(y_test, predicted_labels)

    save_model(model, "models/mnist_cnn_model.h5")
    print("Model saved as models/mnist_cnn_model.h5")
