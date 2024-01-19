from flask import Flask, render_template #Flask framework, renderizar html
import cv2 #OpenCV para procesamiento de imágenes
import numpy as np #Matrices y operaciones

# Crear Flask App
app = Flask(__name__)

# Lista de rutas de imágenes
image_paths = [
    'static/images/urus.jpg',
    'static/images/yz.jpg',
    'static/images/bike.jpg',
    'static/images/sonic.jpg'
]

# Index to keep track of the current image
current_image_index = 0

# Ruta principal
@app.route('/')
def index():
    global current_image_index

    # imread: Lecutra de la imágen.
    selected_image = cv2.imread(image_paths[current_image_index])

    # cvtColor: Convierte la imagen, COLOR_BGR2HSV: formato HSV.
    hsv_image = cv2.cvtColor(selected_image, cv2.COLOR_BGR2HSV)

    # Rango de colores azules en formato Color, Saturación y luminancia.
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])

    # inRange: Crea una máscara para los píxeles azules con los rangos.
    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    # findContours: Encuentra los contornos, RETR_EXTERNAL: Contornos externos que definen los bordes de la blue_mask, CHAIN_APPROX_SIMPLE: Reduce la cantidad de puntos
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Encuentra el objeto azul más grande en la imágen
    largest_contour = max(contours, key=cv2.contourArea)

    # Encuentra el círculo mínimo que encierra el contorno
    center, radius = cv2.minEnclosingCircle(largest_contour)

    # Aproxima un polígono de 5 lados (pentágono) alrededor del círculo
    pentagon = cv2.approxPolyDP(cv2.convexHull(largest_contour), 5, True)

    # Guarda solo las coordenadas del polígono, flatten: convierte a matriz unidimensional, [x1, y1, x2, y2, ..., xn, yn] y tolist(): Convierte la anterior matriz a una lista.
    pentagon_coords = pentagon.flatten().tolist()

    # Guarda la imagen resultante sin el trazado del polígono
    result_image_path = 'static/images/result_image.jpg'
    cv2.imwrite(result_image_path, selected_image)

    # Actualiza el índice por la siguiente imágen
    current_image_index = (current_image_index + 1) % len(image_paths)

    # Renderiza index.html, ruta de la imágen resultante y las coordenadas del pentágono.
    return render_template('index.html', image_path=result_image_path, pentagon_coords=pentagon_coords)

if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')
