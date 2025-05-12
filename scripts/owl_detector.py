#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from nanoowl.owl_predictor import OwlPredictor
import numpy as np
from PIL import Image as PILImage
import cv2
import time
import os
from ament_index_python.packages import get_package_share_directory


class NanoOWLDetectionNode(Node):
    def __init__(self):
        super().__init__('nanoowl_detector_node')
        
        # ROS 2 Subscriptions
        self.subscription = self.create_subscription(
            Image,
            'stitched_image',
            self.listener_callback,
            10
        )
        self.query_subscription = self.create_subscription(
            String,
            'input_query',
            self.query_listener_callback,
            10
        )
        
        # ROS 2 Publishers
        self.image_publisher = self.create_publisher(
            Image,
            'stitched_image_annotated',
            10
        )
        self.detections_publisher = self.create_publisher(
            Detection2DArray,
            'output_detections',
            10
        )

        # Bridge
        self.bridge = CvBridge()

        # 🔎 **Carga del modelo TensorRT optimizado**
        package_path = get_package_share_directory('owl_vit_detector')
        model_path = os.path.join(package_path, 'owl_image_encoder_patch32.engine')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontró el motor optimizado en: {model_path}")
        else:
            self.get_logger().info(f"✅ Motor TensorRT encontrado en: {model_path}")
        
        # Inicialización del predictor de NanoOWL
        self.predictor = OwlPredictor(
            "google/owlvit-base-patch32",
            image_encoder_engine=model_path
        )

        # 🚀 **Configuración por defecto**
        self.query = ["a person", "a car", "a bike"]
        self.text_encodings = None  # Se generarán al primer frame
        self.get_logger().info('NanoOWL cargado exitosamente con TensorRT 🚀')

    def query_listener_callback(self, msg):
        """Callback para actualizar la query dinámica desde ROS 2."""
        self.query = [q.strip() for q in msg.data.split(",")]
        self.text_encodings = None  # Forzamos a recalcular los encodings
        self.get_logger().info(f'🔄 Consulta actualizada: {self.query}')

    def listener_callback(self, msg):
        """
        Callback para procesar la imagen y realizar la inferencia.
        """
        start_time = time.time()

        # Conversión del mensaje de ROS 2 a OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        pil_image = PILImage.fromarray(frame)

        # 🚀 Generación de encodings solo si no existen o se cambió la query
        if not self.text_encodings:
            self.text_encodings = self.predictor.encode_text(self.query)
            self.get_logger().info(f"✅ Text encodings generados para: {self.query}")
    
        # 🚀 Inferencia optimizada
        results = self.predictor.predict(
            image=pil_image, 
            text=self.query, 
            text_encodings=self.text_encodings,
            threshold=0.2
        )

        # 🚀 Post-procesado y anotación de la imagen
        annotated_image = frame.copy()
        detections = Detection2DArray()
        detections.header = msg.header

        # 🔄 **Corrección: Desconectar del grafo de computación**
        labels = results.labels.detach().cpu().numpy()
        scores = results.scores.detach().cpu().numpy()
        boxes = results.boxes.detach().cpu().numpy()

        for idx, label_id in enumerate(labels):
            # Verificamos que el ID no exceda el tamaño de la lista
            if label_id < len(self.query):
                label = self.query[label_id]
            else:
                label = "Unknown"
            
            score = scores[idx]
            bbox = boxes[idx]

            # Dibujar en la imagen
            cv2.rectangle(annotated_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            cv2.putText(annotated_image, f'{label}: {score:.2f}', (int(bbox[0]), int(bbox[1]) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Crear mensaje para ROS 2
            detection = Detection2D()
            detection.bbox.size_x = float(bbox[2] - bbox[0])
            detection.bbox.size_y = float(bbox[3] - bbox[1])
            detection.bbox.center.position.x = float((bbox[0] + bbox[2]) / 2)
            detection.bbox.center.position.y = float((bbox[1] + bbox[3]) / 2)

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = label
            hypothesis.hypothesis.score = float(score)

            detection.results.append(hypothesis)
            detections.detections.append(detection)

        # Publicar detecciones y la imagen anotada
        self.detections_publisher.publish(detections)
        annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='rgb8')
        annotated_msg.header = msg.header
        self.image_publisher.publish(annotated_msg)

        end_time = time.time()
        self.get_logger().info(f"Inferencia completada en {end_time - start_time:.3f} segundos")

def main(args=None):
    rclpy.init(args=args)
    node = NanoOWLDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
