## My plan:

* Use my phone's camera to record a high quality 4k 60 FPS (or higher) video.
* Rather than spin the camera around my head, use an office swivel char and rotate 360 degrees
* Use a script to erase the static background, digest the frames, and spit out the point cloud or model



|My idea|Ai Assessment|
|-|-|
|High-quality phone video|A convenient first experiment. Individual frames may have less detail and more compression than still photographs. Motion blur matters more than the advertised resolution.|
|Swivel chair|Viable if your head and torso rotate together. Small rigid movements can be estimated; changing expression, jaw position, or neck posture is much harder.|
|Remove the background|Correct—but provide explicit exclusion masks to reconstruction, rather than merely replacing the background with black.|
|Build niche processing software|Sensible for capture selection, masking, and quality control. Use an existing reconstruction engine underneath.|




## Harness UI:


| Preview                         | What to show                                                                                          | What it helps you decide                                                            |
| ------------------------------- | ----------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| **Photos and selection**        | Thumbnail grid, full-resolution image viewer, selected/rejected status, rejection reason              | Are the images sharp, complete, and reasonably distributed around the head?         |
| **Masks**                       | Adjustable colored overlay on the original image; optional cutout and mask-only views                 | Are the background and chair excluded without losing ears, hair, or chin?           |
| **Alignment and sparse points** | Sparse point cloud with estimated camera positions and viewing directions; highlight unaligned photos | Did the software understand the views as one coherent head?                         |
| **Dense geometry**              | Colored point cloud, adjustable point size, optional confidence filtering where available             | Is there enough surface coverage? Are there floating fragments or doubled surfaces? |
| **Final mesh**                  | Switch between photographic texture, neutral matte shading, and wireframe overlay                     | Does the surface itself look correct, and is the mesh usable?                       |
