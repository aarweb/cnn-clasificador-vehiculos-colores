from keras.models import load_model
from keras.utils import load_img, img_to_array
import numpy as np
import os

CARPETA_PRUEBAS = "./muestras"
UMBRAL_CERTEZA = 0.80
TAMAÑO_IMAGEN_COLORES = 64
TAMAÑO_IMAGEN_VEHICULOS = 128


def cargar_modelos():
    if not (os.path.exists("colores.h5") and os.path.exists("vehiculos.h5")):
        raise FileNotFoundError("No se han encontrado los modelos entrenados.")

    modelo_colores = load_model("colores.h5")
    modelo_vehiculos = load_model("vehiculos.h5")
    return modelo_colores, modelo_vehiculos


def identificar_color(modelo_colores, imagen):
    prediccion = modelo_colores.predict(imagen, verbose=0)
    certeza = prediccion[0].max()
    indice_clase = prediccion[0].argmax()
    colores = ["amarillo", "azul", "blanco", "rojo"]
    return colores[indice_clase], certeza


def identificar_vehiculo(modelo_vehiculos, imagen):
    prediccion = modelo_vehiculos.predict(imagen, verbose=0)
    certeza = prediccion[0][0]
    if certeza >= 0.5:
        return "coche", certeza
    else:
        return "camión", 1 - certeza


modelo_colores, modelo_vehiculos = cargar_modelos()

for nombre_archivo in os.listdir(CARPETA_PRUEBAS):
    ruta_imagen = os.path.join(CARPETA_PRUEBAS, nombre_archivo)

    imagen_color = load_img(
        ruta_imagen, target_size=(TAMAÑO_IMAGEN_COLORES, TAMAÑO_IMAGEN_COLORES)
    )
    imagen_color = np.expand_dims(img_to_array(imagen_color) / 255.0, axis=0)

    imagen_vehiculo = load_img(
        ruta_imagen, target_size=(TAMAÑO_IMAGEN_VEHICULOS, TAMAÑO_IMAGEN_VEHICULOS)
    )
    imagen_vehiculo = np.expand_dims(img_to_array(imagen_vehiculo) / 255.0, axis=0)

    color, certeza_color = identificar_color(modelo_colores, imagen_color)
    vehiculo, certeza_vehiculo = identificar_vehiculo(modelo_vehiculos, imagen_vehiculo)

    resultado = "No sé qué es"
    if certeza_vehiculo >= UMBRAL_CERTEZA:
        resultado = f"Es un {vehiculo}"
    else:
        resultado = "Es algo"

    if certeza_color >= UMBRAL_CERTEZA:
        resultado += f" de color {color}"

    print(
        f"{nombre_archivo}:\n{resultado}\n (Certeza vehículo: {certeza_vehiculo:.4f}, Certeza color: {certeza_color:.4f})\n"
    )
