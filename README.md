## Ultralytics
`ultralytics` is a standalone ROS 2 service package for YOLO inference.

## Purpose
- In charge of pure ultralytics inference
- Reuse loaded models through in-memory cache

## Service Interfaces
### Detection Service
Service name: `/ultralytics/detect`  
Service type: `detector_interfaces/srv/RunUltralyticsDetect`

Request:
1. `model_path`
2. `image` (`sensor_msgs/Image`)
3. `imgsz`

Response:
1. `success`
2. `message`
3. `boxes_xyxy` (flattened float array, 4 values per box)
4. `class_ids`
5. `confidences`

### Segmentation Service
Service name: `/ultralytics/segment`  
Service type: `detector_interfaces/srv/RunUltralyticsSegment`

Request:
1. `model_path`
2. `image` (`sensor_msgs/Image`)
3. `imgsz`

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
make up-ultralytics
```

## Integration with Detector
- `detector` handlers call this service for every inference frame.

## Integration with Pose Estimator
- `pose_estimator` handlers call `/ultralytics/segment` for mask inference.

## License
Copyright (C) 2026 Shang-Yi Yu

This project is licensed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE).

If you use this software over a network, you must make the complete corresponding source code available to users.
