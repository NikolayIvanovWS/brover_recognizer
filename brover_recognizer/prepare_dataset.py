#!/usr/bin/env python3

from pathlib import Path
import random
import shutil

import cv2
import rclpy
from rclpy.node import Node


DATA_DIR = Path.home() / 'brover_recognizer_data'
TRAIN_RATIO = 0.8
RANDOM_SEED = 42


class DatasetPreparer(Node):
    def __init__(self):
        super().__init__('prepare_dataset')

        self.declare_parameter('classes', ['classA', 'classB'])
        self.classes = self._read_classes()

        self.raw_dir = DATA_DIR / 'dataset' / 'raw'
        self.yolo_dir = DATA_DIR / 'dataset' / 'yolo'
        self.images_dir = self.yolo_dir / 'images'
        self.labels_dir = self.yolo_dir / 'labels'

    def run(self):
        self.get_logger().info(f'Папка исходных изображений: {self.raw_dir}')
        self.get_logger().info(f'Папка YOLO-датасета: {self.yolo_dir}')

        class_images = self._collect_images()
        total_images = sum(len(images) for images in class_images.values())
        if total_images == 0:
            self.get_logger().warning(
                'Изображения не найдены. Сначала запустите collect_dataset.'
            )
            return

        self._reset_yolo_dir()
        split_counts = {'train': 0, 'val': 0}

        rng = random.Random(RANDOM_SEED)
        for class_id, class_name in enumerate(self.classes):
            images = list(class_images[class_name])
            rng.shuffle(images)

            train_images, val_images = self._split_images(images)
            self._copy_split('train', class_id, class_name, train_images)
            self._copy_split('val', class_id, class_name, val_images)

            split_counts['train'] += len(train_images)
            split_counts['val'] += len(val_images)
            self.get_logger().info(
                f'{class_name}: train={len(train_images)}, val={len(val_images)}'
            )

        self._write_data_yaml()
        self.get_logger().info(
            'Подготовка завершена: '
            f"train={split_counts['train']}, val={split_counts['val']}"
        )
        self.get_logger().info(f'YOLO config: {self.yolo_dir / "data.yaml"}')

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

    def _reset_yolo_dir(self):
        if self.yolo_dir.exists():
            shutil.rmtree(self.yolo_dir)

        for split in ('train', 'val'):
            (self.images_dir / split).mkdir(parents=True, exist_ok=True)
            (self.labels_dir / split).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _split_images(images):
        if len(images) <= 1:
            return images, []

        train_count = int(len(images) * TRAIN_RATIO)
        train_count = max(1, min(train_count, len(images) - 1))
        return images[:train_count], images[train_count:]

    def _copy_split(self, split, class_id, class_name, images):
        for image_path in images:
            target_name = self._target_image_name(class_name, image_path)
            target_image = self.images_dir / split / target_name
            target_label = self.labels_dir / split / f'{Path(target_name).stem}.txt'

            shutil.copy2(image_path, target_image)
            target_label.write_text(
                f'{class_id} 0.5 0.5 1.0 1.0\n',
                encoding='utf-8',
            )

    @staticmethod
    def _target_image_name(class_name, image_path):
        if image_path.stem.startswith(f'{class_name}_'):
            return image_path.name
        return f'{class_name}_{image_path.name}'

    def _write_data_yaml(self):
        names = ', '.join(f"'{class_name}'" for class_name in self.classes)
        data_yaml = (
            f'path: {self.yolo_dir}\n'
            'train: images/train\n'
            'val: images/val\n'
            f'nc: {len(self.classes)}\n'
            f'names: [{names}]\n'
        )
        (self.yolo_dir / 'data.yaml').write_text(data_yaml, encoding='utf-8')


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
