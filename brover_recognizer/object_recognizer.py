#!/usr/bin/env python3

from pathlib import Path

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


IMAGE_TOPIC = '/camera1/image_raw'
ANNOTATED_IMAGE_TOPIC = '/brover_recognizer/annotated_image'
RESULT_TOPIC = '/brover_recognizer/result'
DATA_DIR = Path.home() / 'brover_recognizer_data'
MODEL_PATH = DATA_DIR / 'models' / 'best.pt'
IMAGE_SIZE = 320
CONFIDENCE = 0.05
CONFIRMED_CONFIDENCE = 0.25
MAX_DETECTIONS = 1
INFERENCE_PERIOD = 2.0


class ObjectRecognizer(Node):
    def __init__(self):
        super().__init__('object_recognizer')

        self.declare_parameter('classes', ['classA', 'classB'])
        self.classes = self._read_classes()
        self.bridge = CvBridge()
        self.latest_image = None
        self.latest_header = None
        self.last_logged_result = None
        self.model = self._load_model()

        self.create_subscription(
            Image,
            IMAGE_TOPIC,
            self.image_callback,
            10,
        )
        self.annotated_image_publisher = self.create_publisher(
            Image,
            ANNOTATED_IMAGE_TOPIC,
            10,
        )
        self.result_publisher = self.create_publisher(String, RESULT_TOPIC, 10)
        self.create_timer(INFERENCE_PERIOD, self.recognize_latest_image)

        self.get_logger().info(f'Распознавание запущено. Топик камеры: {IMAGE_TOPIC}')
        self.get_logger().info(f'Топик видео с рамками: {ANNOTATED_IMAGE_TOPIC}')
        self.get_logger().info(f'Топик результата: {RESULT_TOPIC}')

    def _read_classes(self):
        classes = list(self.get_parameter('classes').value)
        classes = [str(class_name).strip() for class_name in classes]
        classes = [class_name for class_name in classes if class_name]

        if not classes:
            raise ValueError('Parameter "classes" must contain at least one class')

        return classes

    def _load_model(self):
        if not MODEL_PATH.exists():
            self.get_logger().error(
                f'Модель не найдена: {MODEL_PATH}. '
                'Сначала запустите train_model.launch.py.'
            )
            return None

        try:
            from ultralytics import YOLO
        except ImportError as error:
            self.get_logger().error(f'Python-пакет ultralytics не установлен: {error}')
            return None

        self.get_logger().info(f'Загрузка модели: {MODEL_PATH}')
        return YOLO(str(MODEL_PATH))

    def image_callback(self, message):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(
                message,
                desired_encoding='bgr8',
            )
            self.latest_header = message.header
        except Exception as error:
            self.get_logger().warning(f'Не удалось преобразовать изображение: {error}')

    def recognize_latest_image(self):
        if self.model is None or self.latest_image is None:
            return

        image = self.latest_image.copy()

        try:
            results = self.model.predict(
                source=image,
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE,
                max_det=MAX_DETECTIONS,
                device='cpu',
                verbose=False,
            )
        except Exception as error:
            self.get_logger().error(f'Ошибка распознавания: {error}')
            return

        result = results[0]
        result_text, is_confirmed = self._format_result(result)
        if is_confirmed:
            annotated_image = result.plot()
        else:
            annotated_image = image

        self.result_publisher.publish(String(data=result_text))
        annotated_message = self.bridge.cv2_to_imgmsg(
            annotated_image,
            encoding='bgr8',
        )
        if self.latest_header is not None:
            annotated_message.header = self.latest_header

        self.annotated_image_publisher.publish(annotated_message)

        if result_text != self.last_logged_result:
            self.get_logger().info(result_text)
            self.last_logged_result = result_text

    def _format_result(self, result):
        detections = []

        if result.boxes is not None:
            for class_id, confidence in zip(
                result.boxes.cls.tolist(),
                result.boxes.conf.tolist(),
            ):
                class_index = int(class_id)
                if 0 <= class_index < len(self.classes):
                    detections.append((self.classes[class_index], confidence))

        if not detections:
            return 'Объекты не найдены', False

        detections.sort(key=lambda item: item[1], reverse=True)
        class_name, confidence = detections[0]
        result = f'{class_name}: {confidence:.0%}'

        if confidence < CONFIRMED_CONFIDENCE:
            return f'Нет уверенного распознавания (гипотеза: {result})', False

        return f'Найдено: {result}', True


def main(args=None):
    rclpy.init(args=args)
    node = ObjectRecognizer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Распознавание остановлено')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
