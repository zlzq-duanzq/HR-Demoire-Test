# HR Demoire (test)
## Abstruct 
A morie pattern is an interference pattern produced by overlaying similar but slightly offset templates. Such patterns can be easily generated and observed in reality. For instance, taking a photo of a screen may produce moire pattern on the photo, which can be typically observed as irregular colorful wave-like strips. Moire effect generated accidentally may harm the quality of initial image. Therefore, removing the moire pattern is of great importance in image processing. Recently, many algorithms based on CNN have been proposed, while the majority of them only focused on low-resolution images. However, since high-resolution images and videos become dominant in daily life, this task needs to be completed effectively and efficiently.

In this paper, we designed a deep neural network to remove the moire pattern in high resolution. 
We proposed a image processing pipeline: Downsample -> Demoire -> Super Resolution and Details Restoring from high-resolution. Quantitative and qualitative experiments demonstrate the superiority of our High Resolution Demoire Network (HRDeNet) to the state-of-the-art.



## Qualitative evaluation

![Compare](https://raw.githubusercontent.com/zlzq-duanzq/HR-Demoire-Test/main/web_image/Compare.png)

## Models Structure

### 1. overview
![overall](https://raw.githubusercontent.com/zlzq-duanzq/HR-Demoire-Test/main/web_image/overall%20model.png)

### 2. netEdge
![edge_model](https://raw.githubusercontent.com/zlzq-duanzq/HR-Demoire-Test/main/web_image/edge_model.png)

### 3. netD
![GDN_model](https://raw.githubusercontent.com/zlzq-duanzq/HR-Demoire-Test/main/web_image/GDN_model.png)

### 4. netMerge
![netMerge](https://raw.githubusercontent.com/zlzq-duanzq/HR-Demoire-Test/main/web_image/netMerge.png)

## Test
**To test images**, run

```python
python test.py --dataroot "media/test"  --netGDN "ckpt/netGDN.pth" --netEdge "ckpt/netEdge.pth" --netMerge "ckpt/netMerge.pth" --batchSize 1 --originalSize_h 1080 --originalSize_w 1920 --imageSize_h 1080 --imageSize_w 1920 --image_path "results" --write 1 --record "results.txt"
```

where --dataroot is the root path of test images



**To test video**, run

```python
python test_Video.py --netGDN "ckpt/netGDN.pth" --netEdge "ckpt/netEdge.pth" --netMerge "ckpt/netMerge.pth" --video_path "test_video.avi" --save_path "output_video.avi" --imageSize_h 1080 --imageSize_w 1920
```

where --video_path is the path of the input video, and --save_path is the path of the output (demoired) video. You can monitor the process of demoiring with "current_frame.png"

