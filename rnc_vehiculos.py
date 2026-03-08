from keras.models import Sequential
from keras.layers import (
    Rescaling,
    RandomFlip,
    RandomRotation,
    RandomZoom,
    RandomTranslation,
)
from keras.utils import image_dataset_from_directory
import keras
from recursos.modelos import crear_modelos
from keras.callbacks import EarlyStopping

keras.utils.set_random_seed(42)

CARPETA_VEHICULOS = "./imagenes_practica_vehiculos_colores/vehiculos"
NOMBRE_MODELO = "vehiculos.h5"
RESULTADOS_ENTRENAMIENTO = "resultados_vehiculos.txt"

tamaño_imagen = 128
tamaño_batch = 16
carpeta_train = CARPETA_VEHICULOS + "/train"
carpeta_test = CARPETA_VEHICULOS + "/test"
n_modelos_a_probar = 4


train_dataset = image_dataset_from_directory(
    carpeta_train,
    image_size=(tamaño_imagen, tamaño_imagen),
    batch_size=tamaño_batch,
    labels="inferred",
    label_mode="binary",
    class_names=["camion", "coche"],  # Indico manualmente el orden de las clases
)

test_dataset = image_dataset_from_directory(
    carpeta_test,
    image_size=(tamaño_imagen, tamaño_imagen),
    batch_size=tamaño_batch,
    labels="inferred",
    label_mode="binary",
    class_names=["camion", "coche"],
)

data_augmentation = Sequential(
    [
        RandomFlip("horizontal"),
        RandomRotation(0.2),
        RandomZoom(0.1),
        RandomTranslation(0.1, 0.1),
    ]
)

normalizacion = Rescaling(1.0 / 255)

train_dataset = train_dataset.map(
    lambda x, y: (data_augmentation(normalizacion(x), training=True), y)
)

test_dataset = test_dataset.map(lambda x, y: (normalizacion(x), y))

modelos = crear_modelos(n_modelos_a_probar, tamaño_imagen)
mejor_modelo = None
mejor_acurracy = 0
mejor_loss = float("inf")

with open(RESULTADOS_ENTRENAMIENTO, "w") as archivo_resultados:
    for modelo in modelos:
        modelo.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

        early_stop = EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        )
        modelo.fit(
            train_dataset,
            epochs=50,
            validation_data=test_dataset,
            verbose=0,
            callbacks=[early_stop],
        )

        loss, accuracy = modelo.evaluate(test_dataset, verbose=0)
        linea_resultado = f"{modelo.name};{accuracy:.4f};{loss:.4f}\n"
        print(f"{modelo.name} - Precision: {accuracy:.4f} | Perdida: {loss:.4f}")
        archivo_resultados.write(linea_resultado)

        if accuracy > mejor_acurracy or (
            accuracy == mejor_acurracy and loss < mejor_loss
        ):
            mejor_acurracy = accuracy
            mejor_loss = loss
            mejor_modelo = modelo


print(f"Mejor modelo: {mejor_modelo.name} con precisión {mejor_acurracy:.4f}")
mejor_modelo.save(NOMBRE_MODELO)
