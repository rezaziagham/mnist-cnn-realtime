from keras.models import load_model

def train_model(model, x_train, y_train, epochs=5, batch_size=32, validation_split=0.1):
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=2
    )
    return history

def evaluate_model(model, x_test, y_test):
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy: {acc:.4f}")
    return acc
