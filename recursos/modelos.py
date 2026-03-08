from keras.models import Sequential
from keras.layers import (
    Dropout,
    Input,
    Dense,
    Conv2D,
    MaxPooling2D,
    Flatten,
    BatchNormalization,
)


def crear_modelos(numero_modelos, tamaño_imagen, numero_clases=1):
    modelos = []

    neuronas_densas = [n for n in range(64, 64 * (numero_modelos + 1), 64)]

    activacion_salida = "sigmoid" if numero_clases == 1 else "softmax"

    for i in range(numero_modelos):
        nombre = f"modelo_{i + 1}"
        modelo = Sequential(name=nombre)
        modelo.add(Input(shape=(tamaño_imagen, tamaño_imagen, 3)))
        modelo.add(Conv2D(32, (3, 3), activation="relu"))
        modelo.add(BatchNormalization())
        modelo.add(MaxPooling2D((2, 2)))
        modelo.add(Dropout(0.2))

        # Nos permite detectar patrones segun vayamos profundizando en la red
        for j in range(i):
            modelo.add(Conv2D(64 * (j + 1), (3, 3), activation="relu"))
            modelo.add(BatchNormalization())
            modelo.add(MaxPooling2D((2, 2)))
            modelo.add(Dropout(0.2))

        modelo.add(Flatten())

        modelo.add(Dense(neuronas_densas[i], activation="relu"))
        modelo.add(Dropout(0.2))
        modelo.add(Dense(numero_clases, activation=activacion_salida))

        modelos.append(modelo)

    return modelos
