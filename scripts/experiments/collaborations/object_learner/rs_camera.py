## License: Apache 2.0. See LICENSE file at: https://github.com/IntelRealSense/librealsense/blob/master/LICENSE
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import time
import pyrealsense2 as rs
import numpy as np
import cv2
import bayes3d as b
import os
import shutil
import pickle as pkl
from typing import Union, List


class RSCamera:
    def __init__(self, datadir: str = "./data"):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        if device_product_line == 'L500':
            self.config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        self.datadir = datadir

    def start(self):
        # Start streaming
        cfg = self.pipeline.start(self.config)
        profile = cfg.get_stream(rs.stream.depth)
        intr = profile.as_video_stream_profile().get_intrinsics()

        fx = float(intr.fx) # Focal length of x
        fy = float(intr.fy) # Focal length of y
        ppx = float(intr.ppx) # Principle Point Offsey of x (aka. cx)
        ppy = float(intr.ppy) # Principle Point Offsey of y (aka. cy)
        axs = 0.0 # Axis skew

        self.cam_intrinsics = b.Intrinsics(480,640,fx,fy,ppx,ppy,0.001,255.0)

    def cap(self) -> b.RGBD:

        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Show images
        images = np.hstack((color_image, depth_colormap))
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)

        return b.RGBD(color_image, depth_image, b.identity_pose(), self.cam_intrinsics)

    def stop(self):
        self.pipeline.stop()

    def save_scan(self, num_frames: int, delay_ms: Union[float,None] = 0.0, burnin: int = 10) -> bool:
        if os.path.exists(self.datadir):
            shutil.rmtree(self.datadir)
        os.makedirs(self.datadir)
        i = 0
        try:
            self.start()
            while True:
                if i <= burnin:
                    self.cap()
                    print("BURNIN #%i" % i)
                elif i - burnin <= num_frames:
                    if delay_ms is None:
                        input("Press Enter to cap...")
                    else:
                        pass
                    rgbd = self.cap()
                    with open(self.datadir + "/rgbd_%.5i.pkl" % (i - burnin), 'wb') as fp:
                        pkl.dump(rgbd, fp)
                    print("FRAME #%.5i" % (i - burnin))
                else:
                    break
                i += 1
        finally:
            self.stop()
        return os.listdir(self.datadir) == num_frames

def fetch_images(datadir) -> List[b.RGBD]:
    paths = os.listdir(datadir)
    if len(paths) == 0:
        return None
    paths.sort()
    images = []
    for p in paths:
        with open(datadir + '/' + p, 'rb') as fp:
            rgbd = pkl.load(fp)
            images.append(rgbd)
    return images