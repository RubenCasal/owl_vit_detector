# NanoOWL Detection System

Este proyecto implementa un sistema de detección en tiempo real utilizando **NanoOWL** optimizado con **TensorRT** para inferencias rápidas y eficientes sobre una cámara **Ricoh Theta Z1**. El sistema está desarrollado para **ROS 2 Humble** y se divide en dos nodos principales:

1. **ThetaDriver Node (C++)**: Encargado de capturar el stream de la cámara Ricoh Theta y publicarlo en un tópico ROS 2.
2. **NanoOWLDetectionNode (Python)**: Realiza la detección de objetos sobre el stream capturado y publica los resultados en tópicos ROS 2.

---

## 🚀 **Características**

* Detección en tiempo real sobre imágenes panorámicas 360°.
* Optimización con TensorRT para inferencias rápidas.
* Publicación de imágenes anotadas y bounding boxes en ROS 2.
* Soporte para consultas dinámicas de objetos a detectar mediante un tópico ROS 2.

---

## 🛠️ **Instalación**

```bash
# Clonar el repositorio
git clone <url-repositorio>
cd <nombre-carpeta>

# Instalar dependencias
rosdep install --from-paths src --ignore-src -r -y
colcon build

# Configurar el entorno
source install/setup.bash
```

---

## ⚙️ **Configuración**

El nodo de detección se configura mediante parámetros en ROS 2:

* `use4k`: Activa o desactiva el modo 4K de la cámara.
* `serial`: Especifica el número de serie de la cámara Ricoh Theta Z1.
* `camera_frame`: Define el nombre del frame para las imágenes publicadas.

```bash
ros2 run owl_vit_detector theta_driver --ros-args -p use4k:=true -p serial:=<número-de-serie> -p camera_frame:=camera_1
```

---

## ▶️ **Ejecución**

1️⃣ Lanzar el nodo de la cámara:

```bash
ros2 run owl_vit_detector theta_driver
```

2️⃣ Lanzar el nodo de detección:

```bash
ros2 run owl_vit_detector nanoowl_detector_node
```

3️⃣ Cambiar la consulta (query) en tiempo real:

```bash
ros2 topic pub /input_query std_msgs/String "data: 'a person, a car, a bike'"
```

Puedes especificar los objetos que quieres detectar separándolos por comas.

1️⃣ Lanzar el nodo de la cámara:

```bash
ros2 run owl_vit_detector theta_driver
```

2️⃣ Lanzar el nodo de detección:

```bash
ros2 run owl_vit_detector nanoowl_detector_node
```

---

## 🖼️ **Tópicos Publicados**

* `/stitched_image`: Imagen panorámica capturada.
* `/stitched_image_annotated`: Imagen anotada con las detecciones.
* `/output_detections`: Lista de detecciones en formato ROS 2.

---

## 📦 **Arquitectura del Proyecto**

```
📦 owl_vit_detector
├── include
│   └── owl_vit_detector
│       └── theta_driver_lib.hpp
├── src
│   ├── theta_driver.cpp
│   └── nanoowl_detector_node.py
├── models
│   └── owl_image_encoder_patch32.engine
├── launch
│   └── detection_launch.py
├── CMakeLists.txt
└── package.xml
```

---

## 🤖 **Tecnologías Utilizadas**

* ROS 2 Humble
* TensorRT
* OpenCV
* GStreamer
* NanoOWL
* Ricoh Theta Z1

---

## 🤝 **Contribución**

Si quieres contribuir, realiza un fork del proyecto, crea un branch para tus cambios y abre un PR detallando tus modificaciones.

---

## 📄 **Licencia**

Este proyecto está bajo la licencia MIT.
