## Ultralytics
`ultralytics` is a standalone ROS 2 package used by `detector` for YOLO inference.

## Purpose
- In charge of pure ultralytics inference
- Reuse loaded models through in-memory cache

## Interfaces
### Detection Stream Action
Action name: `/ultralytics/detect_stream`  
Action type: `detector_interfaces/action/RunUltralyticsStream`

Goal:
1. `model_paths`
2. `fps`

The stream action does not accept a confidence threshold. Detection confidence
filtering inside Ultralytics uses the library/model default unless this package
is extended with an explicit inference threshold goal field.

Result:
1. `success`
2. `message`
3. `frames_processed`

Feedback:
1. `frames_processed`
2. `elapsed`

Topic output type: `detector_interfaces/msg/UltralyticsDetections`
1. `stamp`
2. `goal_id`
3. `model_path`
4. `frame_seq`
5. `boxes_xyxy` (flattened float array, 4 values per box)
6. `class_ids`
7. `confidences`

Topic input type: `sensor_msgs/msg/Image` (configured by `input_image_topic`)

### Segmentation Service
Service name: `/ultralytics/segment`  
Service type: `detector_interfaces/srv/RunUltralyticsSegment`

Request:
1. `model_path`
2. `imgsz`

Response:
1. `success`
2. `message`
3. `boxes_xyxy` (flattened float array, 4 values per box)
4. `class_ids`
5. `confidences`
6. `masks_data` (flattened uint8 binary masks)
7. `masks_count`
8. `mask_height`
9. `mask_width`

## Runtime Behavior
- Model loading:
  - cache models by `model_path`
  - cache key is task-aware (`detect:<model_path>` / `segment:<model_path>`)
  - call `fuse()` for non-ONNX/non-TensorRT models
  - warm up model with a dummy image
- Cache cleanup:
  - clear all cached models after 30 seconds without model resolve activity
  - call `torch.cuda.empty_cache()` when CUDA is available

## Run
Using compose (desktop stack):
```
# Select environment
make load-desktop
# or
make load-jetson

make build
make up-detector
```

## Integration with Detector
- `detector` handlers start one stream action per detector action goal.
- Actual box/class/conf outputs are delivered on the detections topic.

## Integration with Pose Estimator
- `pose_estimator` handlers call `/ultralytics/segment` for mask inference.
- Segmentation uses the latest image from `input_image_topic` and returns boxes/classes/confidences/masks in the service response.

## License
Copyright (C) 2026 Shang-Yi Yu, Streamline Bio

This project is licensed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE).

If you use this software over a network, you must make the complete corresponding source code available to users.
