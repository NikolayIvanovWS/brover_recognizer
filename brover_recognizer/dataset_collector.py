#!/usr/bin/env python3

from pathlib import Path
import re
import select
import sys
import termios
import tty

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


IMAGE_TOPIC = '/camera1/image_raw'
DATA_DIR = Path.home() / 'brover_recognizer_data'
IMAGE_SIZE = (640, 480)


class DatasetCollector(Node):
    def __init__(self):
        super().__init__('dataset_collector')

        self.declare_parameter('classes', ['classA', 'classB'])
        self.classes = self._read_classes()
        self.raw_dir = DATA_DIR / 'dataset' / 'raw'
        self.bridge = CvBridge()
        self.current_image = None
        self.saved_counts = {class_name: 0 for class_name in self.classes}

        self._ensure_dirs()
        self.create_subscription(
            Image,
            IMAGE_TOPIC,
            self.image_callback,
            10,
        )

        self.get_logger().info(
            f'Сбор датасета запущен. Топик камеры: {IMAGE_TOPIC}'
        )
        self.get_logger().info(f'Папка данных: {DATA_DIR}')
        self.print_controls()

    def _read_classes(self):
        classes = list(self.get_parameter('classes').value)
        classes = [str(class_name).strip() for class_name in classes]
        classes = [class_name for class_name in classes if class_name]

        if not classes:
            raise ValueError('Параметр "classes" должен содержать хотя бы один класс')

        if len(classes) > 9:
            raise ValueError('Сборщик датасета поддерживает до 9 классов с клавиатуры')

        duplicates = sorted({
            class_name for class_name in classes
            if classes.count(class_name) > 1
        })
        if duplicates:
            raise ValueError(f'Повторяющиеся имена классов: {duplicates}')

        return classes

    def _ensure_dirs(self):
        for class_name in self.classes:
            (self.raw_dir / class_name).mkdir(parents=True, exist_ok=True)

    def image_callback(self, message):
        try:
            image = self.bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
        except Exception as error:
            self.get_logger().warning(f'Не удалось преобразовать изображение: {error}')
            return

        if image is None:
            self.get_logger().warning('Получено пустое изображение')
            return

        if image.shape[1] != IMAGE_SIZE[0] or image.shape[0] != IMAGE_SIZE[1]:
            image = cv2.resize(image, IMAGE_SIZE)

        self.current_image = image

    def print_controls(self):
        print()
        print('Управление сбором датасета:')
        for index, class_name in enumerate(self.classes, start=1):
            print(f'  [{index}] сохранить изображение в {class_name}')
        print('  [q] выйти')
        print()
        sys.stdout.flush()

    def save_image(self, class_name):
        if self.current_image is None:
            self.get_logger().warning(
                'Изображение еще не получено. Проверьте камеру и ROS-топик.'
            )
            return

        class_dir = self.raw_dir / class_name
        file_index = self._next_index(class_dir, class_name)
        file_path = class_dir / f'{class_name}_{file_index:04d}.jpg'

        if not cv2.imwrite(str(file_path), self.current_image):
            self.get_logger().error(f'Не удалось сохранить изображение: {file_path}')
            return

        self.saved_counts[class_name] += 1
        self.get_logger().info(
            f'[{class_name}] сохранено за сеанс: '
            f'{self.saved_counts[class_name]} | {file_path}'
        )

    @staticmethod
    def _next_index(class_dir, class_name):
        max_index = 0
        regexp = re.compile(rf'^{re.escape(class_name)}_(\d+)\.jpg$')

        for path in class_dir.glob(f'{class_name}_*.jpg'):
            match = regexp.match(path.name)
            if not match:
                continue
            max_index = max(max_index, int(match.group(1)))

        return max_index + 1


class KeyboardReader:
    def __init__(self):
        self.tty_file = None
        self.old_settings = None

    def __enter__(self):
        self.tty_file = open('/dev/tty', 'r', encoding='utf-8')
        self.old_settings = termios.tcgetattr(self.tty_file.fileno())
        tty.setcbreak(self.tty_file.fileno())
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.tty_file is not None and self.old_settings is not None:
            termios.tcsetattr(
                self.tty_file.fileno(),
                termios.TCSADRAIN,
                self.old_settings,
            )
        if self.tty_file is not None:
            self.tty_file.close()

    def read_key(self, timeout=0.05):
        ready, _, _ = select.select([self.tty_file], [], [], timeout)
        if not ready:
            return None
        return self.tty_file.read(1)


def main(args=None):
    rclpy.init(args=args)
    node = DatasetCollector()

    try:
        with KeyboardReader() as keyboard:
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.05)
                key = keyboard.read_key(timeout=0.0)

                if key is None:
                    continue
                if key == 'q':
                    node.get_logger().info('Сбор датасета завершен пользователем')
                    break
                if key.isdigit():
                    index = int(key) - 1
                    if 0 <= index < len(node.classes):
                        node.save_image(node.classes[index])
                    else:
                        node.get_logger().warning(f'Нет класса для клавиши: {key}')
    except KeyboardInterrupt:
        node.get_logger().info('Сбор датасета прерван')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
