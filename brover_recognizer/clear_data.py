#!/usr/bin/env python3

from pathlib import Path
import shutil

import rclpy
from rclpy.node import Node


DATA_DIR = Path.home() / 'brover_recognizer_data'
DATASET_DIR = DATA_DIR / 'dataset'
RAW_DIR = DATASET_DIR / 'raw'
MODELS_DIR = DATA_DIR / 'models'


class DataCleaner(Node):
    def __init__(self):
        super().__init__('clear_data')

        self.declare_parameter('classes', ['classA', 'classB'])
        self.classes = self._read_classes()

    def run(self):
        self.get_logger().info(f'Папка данных: {DATA_DIR}')

        for path in (DATASET_DIR, MODELS_DIR):
            if path.exists():
                shutil.rmtree(path)
                self.get_logger().info(f'Удалено: {path}')
            else:
                self.get_logger().info(f'Папка уже отсутствует: {path}')

        for class_name in self.classes:
            class_dir = RAW_DIR / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            self.get_logger().info(f'Создана папка класса: {class_dir}')

        self.get_logger().info('Очистка данных завершена')

    def _read_classes(self):
        classes = list(self.get_parameter('classes').value)
        classes = [str(class_name).strip() for class_name in classes]
        classes = [class_name for class_name in classes if class_name]

        if not classes:
            raise ValueError('Параметр "classes" должен содержать хотя бы один класс')

        duplicates = sorted({
            class_name for class_name in classes
            if classes.count(class_name) > 1
        })
        if duplicates:
            raise ValueError(f'Повторяющиеся имена классов: {duplicates}')

        return classes


def main(args=None):
    rclpy.init(args=args)
    node = DataCleaner()

    try:
        node.run()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
