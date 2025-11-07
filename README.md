# EGH400 - Azure Kinect

Welcome! This guide is designed to help EGH400 students quickly set up and use the Azure Kinect for research. For additional help see the official documentation https://learn.microsoft.com/en-us/previous-versions/azure/kinect-dk

Before starting, the Azure Kinect can technically be powered purely from the USB-A to USB-C connector plugged directly into a computer and used for data transmission and power.
This however will only provide power to the RGB camera and won't allow proper interaction or initialisation of the camera object.
As such to operate properly the Kinect requires power additionally from another source using its power cable.
A few solutions for this were tested:

- **Wall power:**
  This is the most obvious and reliable solution and will require either an alternative adapter to the one packaged, or an extra adapter to change the packaged adapters plug type.

- **PC USB port**:  
  I wasn't able to get enough additional power from pc ports likely due to a lack of supplied current. Feel free to give it a go as it may be useful to have the kinect powered from a laptop but as I said I personally didn't see success from this approach.

- **Power bank**:  
  This is the solution that I eventually landed on as i required the camera to be mobile. If mobility isn't a requirement for your particular research then feel free to use the first solution. It was noted that not all powerbanks provided suffient power, my supervisor Will's powerbank that he had on hand didn't quite work and caused error messages stemming from the IMU to constantly be returned, but my personal powerbank and another fellow students seemed to work to teh same standard as wall power and produced no observed errors.
  
Another quick mention is that early on I noticed alot of tearing in the feed from the camera, I initially thought this may have been a power issue but was able to fix this by capping the cameras framerate at 15 instead of 30 when initialising in code. Not entirely sure why this happens as the camera is designed to work at 30fps, but unless more than 15 fps is needed I found this solution to be appropriate and didnt investigate further.

## Code Setup

The language I selected for my development was Python, to help with this i used the pyk4a library, a wrapper for the official Azure Kinect SDK. This wrapper can be found at https://github.com/etiennedub/pyk4a and contains a lot of useful functions for coding with the azure kinect in python.

With this wrapper you can initialise the Azure Kinect as an object in your code, an example of how to do this is shown below.

```python
from pyk4a import PyK4A

k4a = PyK4A(
    Config(
        color_resolution=pyk4a.ColorResolution.RES_720P,
        depth_mode=pyk4a.DepthMode.NFOV_2X2BINNED,
        synchronized_images_only=True,
        camera_fps=pyk4a.FPS.FPS_15
    )
)
k4a.start()

#Insert Process Code

k4a.stop()
```

## Configuration

- **Color Resolution:**
  The RGB camera sensor includes a range of different resolution options which can be specified in the config as one of the following names or IDs

    OFF = 0,
    RES_720P = 1,
    RES_1080P = 2,
    RES_1440P = 3,
    RES_1536P = 4,
    RES_2160P = 5,
    RES_3072P = 6
<img width="771" height="245" alt="image" src="https://github.com/user-attachments/assets/37cb07fe-cac0-467a-bd21-1300a7c76f0a" />

<br/><br/>

- **Depth Resolution:**
  Similar can be done for the depth camera

    OFF = 0,
    NFOV_2X2BINNED = 1,
    NFOV_UNBINNED = 2,
    WFOV_2X2BINNED = 3,
    WFOV_UNBINNED = 4,
    PASSIVE_IR = 5
<img width="771" height="210" alt="image" src="https://github.com/user-attachments/assets/71dbb821-6a07-45a2-a893-eb74b79564cf" />

<br/><br/>

- **FPS:**
  The FPS can also be set to either 5, 15, or 30, depending on which camera resolution is selected. As noted earlier I used 15 FPS as 30 FPS seemed to produce errors in the captured images.

- **Other:**
  For more customisation options see the official documentation, examples of others parameters that can be customised include image format, exposure, brightness, contrast, white balance, and other camera parameters. Most of these function completely fine under the default settings and do not necessarily need to be specified as shown in the earlier initialisation example.
  

