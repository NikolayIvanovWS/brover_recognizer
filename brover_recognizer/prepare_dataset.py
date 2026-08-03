#!/usr/bin/env python3

from pathlib import Path
import random
import shutil

import cv2
import rclpy
from rclpy.node import Node


DATA_DIR = Path.home() / 'brover_recognizer_data'
CLASSIFY_DATASET_DIR = DATA_DIR / 'dataset' / 'classification'
TRAIN_RATIO = 0.8
RANDOM_SEED = 42


class DatasetPreparer(Node):
    def __init__(self):
        super().__init__('prepare_dataset')

        self.declare_parameter('classes', ['classA', 'classB'])
        self.classes = self._read_classes()

        self.raw_dir = DATA_DIR / 'dataset' / 'raw'
        self.classify_dir = CLASSIFY_DATASET_DIR

    def run(self):
        self.get_logger().info(f'Папка исходных изображений: {self.raw_dir}')
        self.get_logger().info(
            f'Папка classification-датасета: {self.classify_dir}'
        )

        class_images = self._collect_images()
        total_images = sum(len(images) for images in class_images.values())
        if total_images == 0:
            self.get_logger().warning(
                'Изображения не найдены. Сначала запустите collect_dataset.'
            )
            return

        self._reset_classify_dir()
        split_counts = {'train': 0, 'val': 0}

        rng = random.Random(RANDOM_SEED)
        for class_name in self.classes:
            images = list(class_images[class_name])
            rng.shuffle(images)

            train_images, val_images = self._split_images(images)
            self._copy_split('train', class_name, train_images)
            self._copy_split('val', class_name, val_images)

            split_counts['train'] += len(train_images)
            split_counts['val'] += len(val_images)
            self.get_logger().info(
                f'{class_name}: train={len(train_images)}, val={len(val_images)}'
            )

        self.get_logger().info(
            'Подготовка завершена: '
            f"train={split_counts['train']}, val={split_counts['val']}"
        )
        self.get_logger().info(f'Датасет для обучения: {self.classify_dir}')

    def _read_classes(self):
        classes = list(self.get_parameter('classes').value)
        classes = [str(class_name).strip() for class_name in classes]
        classes = [class_name for class_name in classes if class_name]

        if not classes:
            raise ValueError('Parameter "classes" must contain at least one class')

        duplicates = sorted({
            class_name for class_name in classes
            if classes.count(class_name) > 1
        })
        if duplicates:
            raise ValueError(f'Duplicate class names: {duplicates}')

        return classes

    def _collect_images(self):
        class_images = {}
        for class_name in self.classes:
            class_dir = self.raw_dir / class_name
            if not class_dir.exists():
                self.get_logger().warning(f'Папка класса не найдена: {class_dir}')
                class_images[class_name] = []
                continue

            images = [
                image_path
                for image_path in sorted(class_dir.glob('*.jpg'))
                if self._is_valid_image(image_path)
            ]
            class_images[class_name] = images

        return class_images

    def _is_valid_image(self, image_path):
        image = cv2.imread(str(image_path))
        if image is None:
            self.get_logger().warning(f'Файл не похож на изображение: {image_path}')
            return False
        return True

    def _reset_classify_dir(self):
        if self.classify_dir.exists():
            shutil.rmtree(self.classify_dir)

        for split in ('train', 'val'):
            for class_name in self.classes:
                (self.classify_dir / split / class_name).mkdir(
                    parents=True,
                    exist_ok=True,
                )

    @staticmethod
    def _split_images(images):
        if len(images) <= 1:
            return images, []

        train_count = int(len(images) * TRAIN_RATIO)
        train_count = max(1, min(train_count, len(images) - 1))
        return images[:train_count], images[train_count:]

    def _copy_split(self, split, class_name, images):
        for image_path in images:
            target_name = self._target_image_name(class_name, image_path)
            target_image = self.classify_dir / split / class_name / target_name

            shutil.copy2(image_path, target_image)

    @staticmethod
    def _target_image_name(class_name, image_path):
        if image_path.stem.startswith(f'{class_name}_'):
            return image_path.name
        return f'{class_name}_{image_path.name}'

def main(args=None):
    rclpy.init(args=args)
    node = DatasetPreparer()

    try:
        node.run()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
