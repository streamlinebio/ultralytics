# Copyright (C) 2026 Shang-Yi Yu, Streamline Bio
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
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO

from detector_interfaces.action import RunUltralyticsDetect
from detector_interfaces.msg import UltralyticsDetections
from detector_interfaces.srv import RunUltralyticsSegment


class UltralyticsServiceNode(Node):
    def __init__(self) -> None:
        super().__init__('ultralytics_infer_node')

        self.action_name = self.declare_parameter('action_name', '/ultralytics/detect_stream').value
        self.segment_service_name = self.declare_parameter(
            'segment_service_name',
            '/ultralytics/segment',
        ).value
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
        self._latest_image_version = 0
        self._last_resolve_time = 0.0
        self._cache_ttl_sec = 30.0
        self._image_cb_group = MutuallyExclusiveCallbackGroup()
        self._service_cb_group = MutuallyExclusiveCallbackGroup()
        self._action_cb_group = MutuallyExclusiveCallbackGroup()
        self._timer_cb_group = MutuallyExclusiveCallbackGroup()
        self._stream_lock = threading.Lock()
        self._stream_reserved = False

        self._cache_cleanup_timer = self.create_timer(
            1.0, self._cleanup_model_cache, callback_group=self._timer_cb_group
        )

        self.create_subscription(
            Image, self.input_image_topic, self._image_callback, 10, callback_group=self._image_cb_group
        )
        self.detection_publisher = self.create_publisher(UltralyticsDetections, self.output_topic, 10)
        self.stream_action_server = ActionServer(
            self,
            RunUltralyticsDetect,
            self.action_name,
            execute_callback=self.run_detect_stream,
            goal_callback=self._on_stream_goal,
            cancel_callback=self._on_stream_cancel,
            callback_group=self._action_cb_group,
        )
        self.create_service(
            RunUltralyticsSegment,
            self.segment_service_name,
            self.run_segment,
            callback_group=self._service_cb_group,
        )
        self.get_logger().info(
            f'Ultralytics stream ready on {self.action_name}, image_topic={self.input_image_topic}, '
            f'output_topic={self.output_topic}, device={self.device}, default_imgsz={self.default_imgsz}'
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
        channels = 3
        row_pixel_bytes = msg.width * channels  # valid pixel bytes per row, excluding padding
        expected_data_size = msg.height * msg.step

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
            self._latest_image_version += 1

    def _resolve_model(self, model_path: str, task: str):
        """Get ultralytics yolo model."""
        cache_key = f'{task}:{model_path}'
        with self._model_lock:
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

    def _run_detect_model(self, model_path: str, image: np.ndarray, imgsz: int):
        """Run one YOLO detect inference and return flattened detection arrays."""
        model = self._resolve_model(model_path, task='detect')
        with self._model_lock:
            results = model(image, imgsz=imgsz, device=self.device, verbose=False)[0]

        boxes = getattr(results, 'boxes', None)
        if boxes is None or boxes.conf.numel() == 0:
            return (
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.float32),
            )

        return (
            boxes.xyxy.cpu().numpy().astype(np.float32).reshape(-1),
            boxes.cls.cpu().numpy().astype(np.int32),
            boxes.conf.cpu().numpy().astype(np.float32),
        )

    def _publish_detection(
        self,
        *,
        goal_id,
        model_path: str,
        frame_seq: int,
        boxes_xyxy: np.ndarray,
        class_ids: np.ndarray,
        confidences: np.ndarray,
    ) -> None:
        """Publish one detection result for a stream action goal."""
        detection_msg = UltralyticsDetections()
        detection_msg.stamp = self.get_clock().now().to_msg()
        if goal_id is not None:
            detection_msg.goal_id = goal_id
        detection_msg.model_path = model_path
        detection_msg.frame_seq = int(frame_seq)
        detection_msg.boxes_xyxy = boxes_xyxy.tolist()
        detection_msg.class_ids = class_ids.tolist()
        detection_msg.confidences = confidences.tolist()
        self.detection_publisher.publish(detection_msg)

    def _on_stream_goal(self, goal_request: RunUltralyticsDetect.Goal) -> GoalResponse:
        """Accept one active Ultralytics stream goal at a time."""
        model_paths = [model_path.strip() for model_path in goal_request.model_paths if model_path.strip()]
        if not model_paths:
            self.get_logger().warn('Rejecting Ultralytics stream goal: model_paths is empty')
            return GoalResponse.REJECT

        with self._stream_lock:
            if self._stream_reserved:
                self.get_logger().warn('Rejecting Ultralytics stream goal: another stream is active')
                return GoalResponse.REJECT
            self._stream_reserved = True
            return GoalResponse.ACCEPT

    @staticmethod
    def _on_stream_cancel(_goal_handle) -> CancelResponse:
        """Allow clients to stop an active inference stream."""
        return CancelResponse.ACCEPT

    def run_detect_stream(self, goal_handle) -> RunUltralyticsDetect.Result:
        """Run YOLO detection while the action goal is active and publish results to a topic."""
        request = goal_handle.request
        model_paths = [model_path.strip() for model_path in request.model_paths if model_path.strip()]
        imgsz = self.default_imgsz
        fps = float(request.fps) if request.fps > 0.0 else 15.0
        period_sec = 1.0 / max(fps, 1e-6)
        frames_processed = 0
        last_processed_image_version = 0
        t_start = time.monotonic()
        result = RunUltralyticsDetect.Result()

        self.get_logger().info(
            f'Ultralytics stream started: models={model_paths}, fps={fps:.2f}, imgsz={imgsz}'
        )

        try:
            for model_path in model_paths:
                self._resolve_model(model_path, task='detect')

            while rclpy.ok():
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result.success = False
                    result.message = 'Stream cancelled'
                    return result

                # Ensure inferenced image is new from image topic
                loop_start = time.monotonic()
                with self._image_lock:
                    image_version = self._latest_image_version
                    if self._latest_image is None or image_version == last_processed_image_version:
                        image = None
                    else:
                        image = self._latest_image.copy()

                if image is None:
                    time.sleep(min(0.05, period_sec))
                    continue
                last_processed_image_version = image_version

                frame_seq = frames_processed + 1
                for model_path in model_paths:
                    boxes_xyxy, class_ids, confidences = self._run_detect_model(model_path, image, imgsz)
                    self._publish_detection(
                        goal_id=goal_handle.goal_id,
                        model_path=model_path,
                        frame_seq=frame_seq,
                        boxes_xyxy=boxes_xyxy,
                        class_ids=class_ids,
                        confidences=confidences,
                    )
                frames_processed += 1

                feedback = RunUltralyticsDetect.Feedback()
                feedback.frames_processed = frames_processed
                feedback.elapsed = float(time.monotonic() - t_start)
                goal_handle.publish_feedback(feedback)

                sleep_sec = period_sec - (time.monotonic() - loop_start)
                if sleep_sec > 0.0:
                    time.sleep(sleep_sec)

            goal_handle.abort()
            result.success = False
            result.message = 'ROS shutdown'
            return result

        except Exception as exc:
            self.get_logger().error(f'Ultralytics stream failed for models {model_paths}: {exc}')
            goal_handle.abort()
            result.success = False
            result.message = f'inference error: {exc}'
            return result

        finally:
            result.frames_processed = frames_processed
            with self._stream_lock:
                self._stream_reserved = False
            self.get_logger().info(
                f'Ultralytics stream ended: models={model_paths}, frames={frames_processed}'
            )

    def run_segment(
        self,
        request: RunUltralyticsSegment.Request,
        response: RunUltralyticsSegment.Response,
    ):
        """Run yolo segmentation with the latest subscribed image and return raw masks."""
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
            model = self._resolve_model(model_path, task='segment')
            with self._model_lock:
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
