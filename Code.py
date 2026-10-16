import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# No modifiques esta función
def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def load_data(path_imgs, path_labels):
    # Cargar imágenes y etiquetas
    images = read_idx(path_imgs)
    labels = read_idx(path_labels)

    images=images.astype(np.float32) / 255.0  # Normalizar imágenes a [0, 1]

    # Dividir los datos en entrenamiento y prueba (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        images,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    return (X_train, y_train, X_test, y_test)

path_imgs = r'C:\Users\colla\Coding\jcvenv\TripleTen\Sprint 18\MNIST\mnist\t10k-images.idx3-ubyte'

path_labels = r'C:\Users\colla\Coding\jcvenv\TripleTen\Sprint 18\MNIST\mnist\t10k-labels.idx1-ubyte'

X_train, y_train, X_test, y_test = load_data(path_imgs, path_labels)

print('Dataset shape:', X_train.shape[0], "[n_images,hight,width]")
plt.imshow(X_train[0].reshape(28, 28), cmap='gray')
plt.title(f"Label: {y_train[0]}")
plt.show()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam

def create_model(input_shape, num_classes):
    # Inicializar el modelo secuencial
    model = Sequential()

    # Aplanar las imágenes de 28x28 a un vector de 784 valores
    model.add(Flatten(input_shape=input_shape))

    # Primera capa oculta: 64 neuronas
    model.add(Dense(64, activation='relu'))

    # Segunda capa oculta: 32 neuronas
    model.add(Dense(32, activation='relu'))

    # Tercera capa oculta: 16 neuronas
    model.add(Dense(16, activation='relu'))

    # Capa de salida: 10 clases
    model.add(Dense(num_classes, activation='softmax'))

    # Compilar el modelo
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['acc']
    )

    return model


# Verificar la arquitectura de la red neuronal
create_model((28, 28), 10).summary()


def train_model(model, train_data, test_data, batch_size=32, epochs=5):
    X_train, y_train = train_data
    X_test, y_test = test_data

    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        batch_size=batch_size,
        epochs=epochs,
        verbose=2,
        shuffle=True
    )

    return model

# MNIST file paths
imgs = r'C:\Users\colla\Coding\jcvenv\TripleTen\Sprint 18\MNIST\t10k-images.idx3-ubyte'
labels = r'C:\Users\colla\Coding\jcvenv\TripleTen\Sprint 18\MNIST\t10k-labels.idx1-ubyte'

# Load and split the dataset
X_train, y_train, X_test, y_test = load_data(imgs, labels)

# Create the neural network
model = create_model((28, 28), 10)

# Display the model architecture
model.summary()

# Train the neural network
trained_model = train_model(
    model,
    (X_train, y_train),
    (X_test, y_test),
    batch_size=32,
    epochs=10
)

# Evaluate training and test accuracy
train_loss, train_acc = trained_model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = trained_model.evaluate(X_test, y_test, verbose=0)

print(f'Train accuracy: {train_acc:.4f}')
print(f'Test accuracy: {test_acc:.4f}')