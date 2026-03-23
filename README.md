## Ultralytics
`ultralytics` is a standalone ROS 2 service package used by `detector` for YOLO inference.

## Purpose
- In charge of pure ultralytics inferece
- Reuse loaded models through in-memory cache

## Service Interface
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
make up-ultralytics
```

## Integration with Detector
- `detector` handlers call this service for every inference frame.
