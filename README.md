

### Models

![overall model](C:\Users\86138\Desktop\paper\overall model.png)

### Qualitative evaluation

![Compare](C:\Users\86138\Desktop\paper\Compare.png)

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

