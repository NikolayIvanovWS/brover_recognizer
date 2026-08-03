#!/usr/bin/env python3

from pathlib import Path
import shutil

import rclpy
from rclpy.node import Node


DATA_DIR = Path.home() / 'brover_recognizer_data'
YOLO_DATASET_DIR = DATA_DIR / 'dataset' / 'yolo'
YOLO_DATA_YAML = YOLO_DATASET_DIR / 'data.yaml'
MODELS_DIR = DATA_DIR / 'models'

MODEL_NAME = 'yolo26n.pt'
IMAGE_SIZE = 320
EPOCHS = 30
BATCH_SIZE = 2
WORKERS = 0


class ModelTrainer(Node):
    def __init__(self):
        super().__init__('train_model')

    def run(self):
        if not YOLO_DATA_YAML.exists():
            self.get_logger().error(
                f'YOLO dataset config not found: {YOLO_DATA_YAML}'
            )
            self.get_logger().error(
                'Сначала запустите: '
                'ros2 launch brover_recognizer prepare_dataset.launch.py'
            )
            return

        try:
            from ultralytics import YOLO
        except ImportError:
            self.get_logger().error(
                'Python-пакет ultralytics не установлен.'
            )
            self.get_logger().error(
                'Для обучения YOLO установите зависимости, например: '
                'python3 -m pip install --user "ultralytics[export]"'
            )
            return

        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        self.get_logger().info(f'Модель: {MODEL_NAME}')
        self.get_logger().info(f'Датасет: {YOLO_DATA_YAML}')
        self.get_logger().info(
            'Параметры обучения: '
            f'imgsz={IMAGE_SIZE}, epochs={EPOCHS}, '
            f'batch={BATCH_SIZE}, workers={WORKERS}'
        )

        model = YOLO(MODEL_NAME)
        results = model.train(
            data=str(YOLO_DATA_YAML),
            epochs=EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            workers=WORKERS,
            device='cpu',
            project=str(MODELS_DIR),
            name='train',
            exist_ok=True,
        )

        save_dir = Path(results.save_dir)
        best_model = save_dir / 'weights' / 'best.pt'
        latest_model = MODELS_DIR / 'best.pt'

        if best_model.exists():
            shutil.copy2(best_model, latest_model)
            self.get_logger().info(f'Лучшая модель: {best_model}')
            self.get_logger().info(f'Копия для запуска: {latest_model}')
        else:
            self.get_logger().warning(
                f'Файл лучшей модели не найден: {best_model}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = ModelTrainer()

    try:
        node.run()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
