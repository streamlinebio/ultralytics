## Ultralytics
`ultralytics` is a standalone ROS 2 package used by `detector` for YOLO inference.

## Purpose
- In charge of pure ultralytics inferece
- Reuse loaded models through in-memory cache

## Interfaces
Service type: `detector_interfaces/srv/RunUltralyticsDetect`

Request:
1. `model_path`
2. `imgsz`

Response:
1. `success`
2. `message`

Topic output type: `detector_interfaces/msg/UltralyticsDetections`
1. `stamp`
2. `boxes_xyxy` (flattened float array, 4 values per box)
3. `class_ids`
4. `confidences`

Topic input type: `sensor_msgs/msg/Image` (configured by `input_image_topic`)

## Runtime Behavior
- Model loading:
  - cache models by `model_path`
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
- `detector` handlers call the service for each inference trigger.
- Actual box/class/conf outputs are delivered on the detections topic.

## License
Copyright (C) 2026 Shang-Yi Yu

This project is licensed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE).

If you use this software over a network, you must make the complete corresponding source code available to users.
