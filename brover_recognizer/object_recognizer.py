#!/usr/bin/env python3

import os
from pathlib import Path

from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String


os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import torch
from ultralytics import YOLO

IMAGE_TOPIC = '/camera1/image_raw'
RESULT_TOPIC = '/brover_recognizer/result'
DATA_DIR = Path.home() / 'brover_recognizer_data'
MODEL_PATH = DATA_DIR / 'models' / 'best.pt'
IMAGE_SIZE = 320
CONFIRMED_CONFIDENCE = 0.90
INFERENCE_PERIOD = 2.0


class ObjectRecognizer(Node):
    def __init__(self):
        super().__init__('object_recognizer')

        self.declare_parameter('classes', ['classA', 'classB'])
        self.classes = self._read_classes()
        self.bridge = CvBridge()
        self.latest_image_message = None
        self.last_logged_result = None
        self.model = self._load_model()
        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self.create_subscription(
            Image,
            IMAGE_TOPIC,
            self.image_callback,
            image_qos,
        )
        self.result_publisher = self.create_publisher(String, RESULT_TOPIC, 10)
        self.create_timer(INFERENCE_PERIOD, self.recognize_latest_image)

        self.get_logger().info(f'Распознавание запущено. Топик камеры: {IMAGE_TOPIC}')
        self.get_logger().info(f'Топик результата: {RESULT_TOPIC}')

    def _read_classes(self):
        classes = list(self.get_parameter('classes').value)
        classes = [str(class_name).strip() for class_name in classes]
        classes = [class_name for class_name in classes if class_name]

        if not classes:
            raise ValueError('Параметр "classes" должен содержать хотя бы один класс')

        return classes

    def _load_model(self):
        if not MODEL_PATH.exists():
            self.get_logger().error(
                f'Модель не найдена: {MODEL_PATH}. '
                'Сначала запустите train_model.launch.py.'
            )
            return None

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        self.get_logger().info(f'Загрузка модели: {MODEL_PATH}')
        return YOLO(str(MODEL_PATH))

    def image_callback(self, message):
        self.latest_image_message = message

    def recognize_latest_image(self):
        if self.model is None or self.latest_image_message is None:
            return

        try:
            image = self.bridge.imgmsg_to_cv2(
                self.latest_image_message,
                desired_encoding='bgr8',
            )
        except Exception as error:
            self.get_logger().warning(f'Не удалось преобразовать изображение: {error}')
            return

        try:
            results = self.model.predict(
                source=image,
                imgsz=IMAGE_SIZE,
                device='cpu',
                verbose=False,
            )
        except Exception as error:
            self.get_logger().error(f'Ошибка распознавания: {error}')
            return

        result = results[0]
        result_text = self._format_result(result)

        self.result_publisher.publish(String(data=result_text))

        if result_text != self.last_logged_result:
            self.get_logger().info(result_text)
            self.last_logged_result = result_text

    def _format_result(self, result):
        if result.probs is None:
            return 'Объекты не найдены'

        class_index = int(result.probs.top1)
        confidence = float(result.probs.top1conf)
        class_name = self._class_name(class_index)
        result = f'{class_name}: {confidence:.0%}'

        if confidence < CONFIRMED_CONFIDENCE:
            return 'Ничего не распознано'

        return f'Найдено: {result}'

    def _class_name(self, class_index):
        model_names = getattr(self.model, 'names', None)
        if isinstance(model_names, dict) and class_index in model_names:
            return str(model_names[class_index])
        if 0 <= class_index < len(self.classes):
            return self.classes[class_index]
        return f'class{class_index}'


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
