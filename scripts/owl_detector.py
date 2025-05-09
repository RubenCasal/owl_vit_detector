#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import numpy as np
from PIL import Image as PILImage
import cv2
import time


class OWLVitDetectionNode(Node):
    def __init__(self):
        super().__init__('owlvit_detector_node')
        
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

        # Bridge and Model Loading
        self.bridge = CvBridge()
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

        # 🚀 Mover el modelo a GPU si está disponible
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
            self.device = "cuda"
            self.get_logger().info("🚀 OWL-ViT detector cargado en GPU")
        else:
            self.device = "cpu"
            self.get_logger().info("⚠️ OWL-ViT detector cargado en CPU")
        
        # Verificación
        print(f"Modelo en: {next(self.model.parameters()).device}")

        # Default query and logging
        self.query = ["a person", "a car", "a bike"]
        self.get_logger().info('OWL-ViT detector loaded and running.')

    def query_listener_callback(self, msg):
        self.query = [q.strip() for q in msg.data.split(",")]
        self.get_logger().info(f'Updated query: {self.query}')

    def listener_callback(self, msg):
        """
        Callback para procesar la imagen y realizar la inferencia.
        """
        start_time = time.time()

        # Conversión del mensaje de ROS 2 a OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        
        # 🚀 Preprocesado directo con el processor
        pil_image = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        # 🚀 Generación de inputs y movimiento a GPU
        inputs = self.processor(text=[self.query], images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        preproc_time = time.time()
    
        # 🚀 Inferencia directa en GPU
        with torch.no_grad():
            outputs = self.model(**inputs)

     
        
        # 🚀 Post-procesado
        target_sizes = torch.Tensor([pil_image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)

        # 🚀 Optimización: Anotación de imagen solo para detecciones válidas
        annotated_image = frame.copy()
        detections = Detection2DArray()
        detections.header = msg.header

        for box, score, label in zip(results[0]["boxes"], results[0]["scores"], results[0]["labels"]):
            if score > 0.3:  # Solo dibujar si el score es alto
                box = [int(i) for i in box.tolist()]
                label_name = self.query[label]
                cv2.rectangle(annotated_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(annotated_image, f'{label_name}: {score:.2f}', (box[0], box[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Crear mensaje para ROS 2
                detection = Detection2D()
                detection.bbox.size_x = float(box[2] - box[0])
                detection.bbox.size_y = float(box[3] - box[1])
                detection.bbox.center.position.x = float((box[0] + box[2]) / 2)
                detection.bbox.center.position.y = float((box[1] + box[3]) / 2)

                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = label_name
                hypothesis.hypothesis.score = float(score)

                detection.results.append(hypothesis)
                detections.detections.append(detection)

        # Publicar detecciones
        self.detections_publisher.publish(detections)

        # Convertir a mensaje de ROS 2
        annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='rgb8')
        annotated_msg.header = msg.header
        self.image_publisher.publish(annotated_msg)

      

def main(args=None):
    rclpy.init(args=args)
    node = OWLVitDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
