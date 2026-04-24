# Copyright (C) 2026 Shang-Yi Yu
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import os
import threading
import time
from typing import Dict

import numpy as np
import rclpy
import torch
import cv2
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO

from detector_interfaces.msg import UltralyticsDetections
from detector_interfaces.srv import RunUltralyticsDetect


class UltralyticsServiceNode(Node):
    def __init__(self) -> None:
        super().__init__('ultralytics_infer_node')

        self.service_name = self.declare_parameter('service_name', '/ultralytics/detect').value
        self.input_image_topic = self.declare_parameter(
            'input_image_topic', '/camera/realsense/color/image_rect_raw'
        ).value
        self.output_topic = self.declare_parameter('output_topic', '/ultralytics/detection_results').value
        self.default_imgsz = int(self.declare_parameter('default_imgsz', 736).value)
        self.device = self._select_device()

        self._image_lock = threading.Lock()
        self._model_lock = threading.Lock()
        self._model_cache: Dict[str, YOLO] = {}
        self._latest_image: np.ndarray | None = None
        self._last_resolve_time = 0.0
        self._cache_ttl_sec = 30.0
        self._image_cb_group = MutuallyExclusiveCallbackGroup()
        self._service_cb_group = MutuallyExclusiveCallbackGroup()
        self._timer_cb_group = MutuallyExclusiveCallbackGroup()

        self._cache_cleanup_timer = self.create_timer(
            1.0, self._cleanup_model_cache, callback_group=self._timer_cb_group
        )

        self.create_subscription(
            Image, self.input_image_topic, self._image_callback, 10, callback_group=self._image_cb_group
        )
        self.detection_publisher = self.create_publisher(UltralyticsDetections, self.output_topic, 10)
        self.create_service(
            RunUltralyticsDetect, self.service_name, self.run_detect, callback_group=self._service_cb_group
        )
        self.get_logger().info(
            f'Ultralytics service ready on {self.service_name}, image_topic={self.input_image_topic}, '
            f'output_topic={self.output_topic}, device={self.device}, default_imgsz={self.default_imgsz}'
        )

    def _select_device(self) -> str:
        device_env = os.environ.get('YOLO_DEVICE')
        if device_env:
            return device_env
        return 'cuda:0' if torch.cuda.is_available() else 'cpu'

    @staticmethod
    def _imgmsg_to_cv2(msg, desired_encoding='bgr8') -> np.ndarray:
        """Convert from ROS2 msg to cv2 image."""
        channels = 3
        row_pixel_bytes = msg.width * channels # valid pixel bytes per row (excluding padding)
        expected_data_size = msg.height * msg.step # total buffer size including row padding

        # reshape raw buffer into (H, step) to preserve stride/padding
        raw = np.frombuffer(msg.data, dtype=np.uint8, count=expected_data_size).reshape(msg.height, msg.step)
        # remove padding (if any) and reshape into standard (H, W, C) image
        raw_image = raw[:, :row_pixel_bytes].reshape(msg.height, msg.width, channels)

        if msg.encoding == desired_encoding:
            return raw_image

        if msg.encoding == 'rgb8' and desired_encoding == 'bgr8':
            return cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)

        raise RuntimeError(f'Unknown encoding transform: {msg.encoding} -> {desired_encoding}')

    def _image_callback(self, msg: Image) -> None:
        try:
            image = self._imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'Failed to decode input image on {self.input_image_topic}: {exc}')
            return

        with self._image_lock:
            self._latest_image = image.copy()

    def _resolve_model(self, model_path: str):
        """Get ultralytics yolo detection model."""
        with self._model_lock:
            self._last_resolve_time = time.monotonic()
            cached_model = self._model_cache.get(model_path)
            if cached_model is not None:
                return cached_model

            model = YOLO(model_path, task='detect')
            if not model_path.endswith(('.engine', '.onnx')):
                model.fuse()

            if torch.cuda.is_available() and model_path.endswith('.pt'):
                if hasattr(model, 'model') and hasattr(model.model, 'to'):
                    model.model.to(self.device)
                elif hasattr(model, 'to'):
                    model.to(self.device)

            # Model warm-up
            dummy = np.zeros((self.default_imgsz, self.default_imgsz, 3), dtype=np.uint8)
            try:
                _ = model(dummy, imgsz=self.default_imgsz, device=self.device, verbose=False)
            except Exception:
                try:
                    _ = model(dummy, verbose=False)
                except Exception:
                    pass

            self._model_cache[model_path] = model
            self.get_logger().info(f'Loaded model: {model_path}')
            return model

    def _cleanup_model_cache(self) -> None:
        """Release cache model after sepcific duration from last action called."""
        with self._model_lock:
            if not self._model_cache:
                return

            idle_sec = time.monotonic() - self._last_resolve_time
            if idle_sec < self._cache_ttl_sec:
                return

            self._model_cache.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.get_logger().info('Unloaded cached models after 30 seconds of resolve inactivity')

    def run_detect(self, request: RunUltralyticsDetect.Request, response: RunUltralyticsDetect.Response):
        """Run yolo detection with the latest subscribed image and publish outputs to a topic."""
        model_path = request.model_path.strip()
        if not model_path:
            response.success = False
            response.message = 'model_path is empty'
            return response

        with self._image_lock:
            if self._latest_image is None:
                response.success = False
                response.message = f'no image received on topic: {self.input_image_topic}'
                return response
            image = self._latest_image.copy()

        imgsz = int(request.imgsz) if request.imgsz > 0 else self.default_imgsz

        try:
            model = self._resolve_model(model_path)
            with self._model_lock:
                results = model(image, imgsz=imgsz, device=self.device, verbose=False)[0]

            boxes = getattr(results, 'boxes', None)
            if boxes is None or boxes.conf.numel() == 0:
                boxes_xyxy = np.empty((0,), dtype=np.float32)
                classes = np.empty((0,), dtype=np.int32)
                confidences = np.empty((0,), dtype=np.float32)
            else:
                boxes_xyxy = boxes.xyxy.cpu().numpy().astype(np.float32).reshape(-1)
                classes = boxes.cls.cpu().numpy().astype(np.int32)
                confidences = boxes.conf.cpu().numpy().astype(np.float32)

            detection_msg = UltralyticsDetections()
            detection_msg.stamp = self.get_clock().now().to_msg()
            detection_msg.boxes_xyxy = boxes_xyxy.tolist()
            detection_msg.class_ids = classes.tolist()
            detection_msg.confidences = confidences.tolist()
            self.detection_publisher.publish(detection_msg)

            response.success = True
            response.message = 'OK'
            return response
        except Exception as exc:
            response.success = False
            response.message = f'inference error: {exc}'
            self.get_logger().error(f'Inference failed for model {model_path}: {exc}')
            return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = UltralyticsServiceNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
