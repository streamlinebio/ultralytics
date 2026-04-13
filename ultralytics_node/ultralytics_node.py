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
from rclpy.node import Node
from ultralytics import YOLO

from detector_interfaces.srv import RunUltralyticsDetect, RunUltralyticsSegment


class UltralyticsServiceNode(Node):
    def __init__(self) -> None:
        super().__init__('ultralytics_infer_node')

        self.detection_service_name = self.declare_parameter('detection_service_name', '/ultralytics/detect').value
        self.segment_service_name = self.declare_parameter(
            'segment_service_name',
            '/ultralytics/segment',
        ).value
        self.default_imgsz = int(self.declare_parameter('default_imgsz', 736).value)
        self.device = self._select_device()

        self._lock = threading.Lock()
        self._model_cache: Dict[str, YOLO] = {}
        self._last_resolve_time = 0.0
        self._cache_ttl_sec = 30.0
        self._cache_cleanup_timer = self.create_timer(1.0, self._cleanup_model_cache)

        self.create_service(RunUltralyticsDetect, self.detection_service_name, self.run_detect)
        self.create_service(RunUltralyticsSegment, self.segment_service_name, self.run_segment)
        self.get_logger().info(
            f'Ultralytics service ready on {self.detection_service_name}, device={self.device}, default_imgsz={self.default_imgsz}'
        )
        self.get_logger().info(f'Ultralytics segmentation service ready on {self.segment_service_name}')

    def _select_device(self) -> str:
        device_env = os.environ.get('YOLO_DEVICE')
        if device_env:
            return device_env
        return 'cuda:0' if torch.cuda.is_available() else 'cpu'

    @staticmethod
    def _imgmsg_to_cv2(msg, desired_encoding='bgr8') -> np.ndarray:
        """Convert from ROS2 msg to cv2 image."""
        raw_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)

        if msg.encoding == desired_encoding:
            return raw_image

        if msg.encoding == 'rgb8' and desired_encoding == 'bgr8':
            return cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)

        raise RuntimeError(f'Unknown encoding transform: {msg.encoding} -> {desired_encoding}')

    def _resolve_model(self, model_path: str, task: str):
        """Get ultralytics yolo model."""
        cache_key = f'{task}:{model_path}'
        with self._lock:
            self._last_resolve_time = time.monotonic()
            cached_model = self._model_cache.get(cache_key)
            if cached_model is not None:
                return cached_model

            model = YOLO(model_path, task=task)
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

            self._model_cache[cache_key] = model
            self.get_logger().info(f'Loaded model: {model_path} (task={task})')
            return model

    def _cleanup_model_cache(self) -> None:
        """Release cache model after sepcific duration from last action called."""
        with self._lock:
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
        """Run yolo detection given model path."""
        model_path = request.model_path.strip()
        if not model_path:
            response.success = False
            response.message = 'model_path is empty'
            return response

        try:
            image = self._imgmsg_to_cv2(request.image, desired_encoding='bgr8')
        except Exception as exc:
            response.success = False
            response.message = f'invalid image: {exc}'
            return response

        imgsz = int(request.imgsz) if request.imgsz > 0 else self.default_imgsz

        try:
            model = self._resolve_model(model_path, task='detect')
            with self._lock:
                results = model(image, imgsz=imgsz, device=self.device, verbose=False)[0]

            boxes = getattr(results, 'boxes', None)
            if boxes is None or boxes.conf.numel() == 0:
                response.success = True
                response.message = 'OK'
                response.boxes_xyxy = []
                response.class_ids = []
                response.confidences = []
                return response

            boxes_xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
            classes = boxes.cls.cpu().numpy().astype(np.int32)
            confidences = boxes.conf.cpu().numpy().astype(np.float32)

            response.success = True
            response.message = 'OK'
            response.boxes_xyxy = boxes_xyxy.reshape(-1).tolist()
            response.class_ids = classes.tolist()
            response.confidences = confidences.tolist()
            return response
        except Exception as exc:
            response.success = False
            response.message = f'inference error: {exc}'
            self.get_logger().error(f'Inference failed for model {model_path}: {exc}')
            return response

    def run_segment(
        self,
        request: RunUltralyticsSegment.Request,
        response: RunUltralyticsSegment.Response,
    ):
        """Run yolo segmentation and return raw masks."""
        model_path = request.model_path.strip()
        if not model_path:
            response.success = False
            response.message = 'model_path is empty'
            return response

        try:
            image = self._imgmsg_to_cv2(request.image, desired_encoding='bgr8')
        except Exception as exc:
            response.success = False
            response.message = f'invalid image: {exc}'
            return response

        imgsz = int(request.imgsz) if request.imgsz > 0 else self.default_imgsz

        try:
            model = self._resolve_model(model_path, task='segment')
            with self._lock:
                results = model(image, imgsz=imgsz, device=self.device, verbose=False)[0]

            boxes = getattr(results, 'boxes', None)
            masks = getattr(results, 'masks', None)
            if boxes is None or boxes.conf.numel() == 0 or masks is None or masks.data.numel() == 0:
                response.success = True
                response.message = 'OK'
                response.boxes_xyxy = []
                response.class_ids = []
                response.confidences = []
                response.masks_data = []
                response.masks_count = 0
                response.mask_height = 0
                response.mask_width = 0
                return response

            boxes_xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
            classes = boxes.cls.cpu().numpy().astype(np.int32)
            confidences = boxes.conf.cpu().numpy().astype(np.float32)
            masks_np = masks.data.cpu().numpy()
            masks_bin = (masks_np > 0.5).astype(np.uint8)
            mask_count, mask_height, mask_width = masks_bin.shape

            response.success = True
            response.message = 'OK'
            response.boxes_xyxy = boxes_xyxy.reshape(-1).tolist()
            response.class_ids = classes.tolist()
            response.confidences = confidences.tolist()
            response.masks_data = masks_bin.reshape(-1).tolist()
            response.masks_count = int(mask_count)
            response.mask_height = int(mask_height)
            response.mask_width = int(mask_width)
            return response
        except Exception as exc:
            response.success = False
            response.message = f'inference error: {exc}'
            self.get_logger().error(f'Segmentation failed for model {model_path}: {exc}')
            return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = UltralyticsServiceNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
