# FFmpeg → OpenCV/SAM 2 → COLMAP → OpenMVS → Blender



|Stage|Free options|Where an agentic coder helps|
|-|-|-|
|**Extract video frames**|[FFmpeg](https://ffmpeg.org/)|Preserve resolution, handle phone-video orientation, extract timestamps, and produce a reproducible image set.|
|**Select useful frames**|Python + [OpenCV](https://opencv.org/)|Score sharpness within the head region, reject near-duplicates, retain angular coverage, and generate a contact sheet for review.|
|**Mask the background**|OpenCV for simple backgrounds; [SAM 2](https://github.com/facebookresearch/sam2) for prompted video segmentation|Build a “click your head” interface, propagate masks through the video, allow corrections, and export masks in the reconstruction tool’s required format.|
|**Estimate viewpoints and sparse geometry**|[COLMAP / PyCOLMAP](https://colmap.github.io/)|Configure camera parameters, match images, run reconstruction, and diagnose disconnected or incorrectly aligned views.|
|**Build dense geometry**|COLMAP’s multi-view stereo, or [OpenMVS](https://github.com/cdcseacave/openMVS)|Convert formats, manage memory and resolution, run depth estimation, and fuse the results into a dense point cloud.|
|**Create and texture the mesh**|OpenMVS|Automate surface reconstruction, refinement, and photographic texturing.|
|**Clean up and export**|[Blender](https://www.blender.org/), [Open3D](https://www.open3d.org/)|Remove isolated fragments, simplify geometry, set scale, render inspection views, and export suitable files. Artistic repairs may still need your judgment.|



