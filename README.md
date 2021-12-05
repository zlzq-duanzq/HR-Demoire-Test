# HR Demoire (test)
## Abstruct
Moiré effect is a visual perception that occurs when viewing a set of lines or dots that is superimposed on another set of lines or dots, where the sets differ in relative size, angle, or spacing \cite{def_moire_effect}. A morie pattern is an interference pattern produced by overlaying similar but slightly offset templates \cite{def_moire_pattern}. Such patterns can be easily generated and observed in reality. For instance, taking a photo of a screen may produce moire pattern on the photo, which can be typically observed as irregular colorful wave-like strips. Moire effect generated accidentally may harm the quality of initial image. Therefore, removing the moire pattern is of great importance in image processing. Since moire pattern's shape and position on a photo is generally irregular, it can hardly be expressed as some analytical model \cite{hefhde2net}. Therefore, although the moire pattern is obvious for human, designing a rule-based algorithm to assist a computer to recognize it is difficult. Due to the nature of moire pattern, removing it remains a tricky problem. 


 Previous moire pattern removing algorithms generally use filters in frequency domain \cite{newmethod}. However, the disadvantage of filters is that they can smooth details of the target image. Recently, with the development of deep neural network, many algorithms based on CNN have been proposed. Compared with conventional algorithms, they have better performance. In this paper, we propose a new deep neural network based algorithm to solve the moire pattern removing problem.
 
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

