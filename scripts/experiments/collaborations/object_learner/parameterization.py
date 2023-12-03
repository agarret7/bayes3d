import numpy as np
import jax.numpy as jnp
import jax
import bayes3d as b
import time
from PIL import Image
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import cv2
import trimesh
import os
import glob
import bayes3d.neural
from bayes3d.colmap.colmap_utils import geom_transform_points
from bayes3d.utils.icp import find_least_squares_transform_between_clouds

import pickle
# Can be helpful for debugging:
# jax.config.update('jax_enable_checks', True)
from bayes3d.neural.segmentation import carvekit_get_foreground_mask
from tqdm import tqdm

from rs_camera import RSCamera, fetch_images
from tools import ContactParamEnumerator

import random

class Normal:
    @staticmethod
    def logpdf(x: float, mean: float, stdev: float) -> float:
        # TODO: make this differentiable with Torch or Jax
        z = (x - mean) / stdev
        return -(abs(z)**2 + np.log(2.0*np.pi))/2.0 - np.log(stdev)

    @staticmethod
    def random(mean: float, stdev: float) -> float:
        u = random.random() * 2.0 - 1.0
        v = random.random() * 2.0 - 1.0
        r = u**2 + v**2
        if r == 0.0 or r > 1.0:
            return Normal.random(mean, stdev)
        c = (-2.0 * np.log(r) / r) ** 0.5
        return u * c * stdev + mean

def learn_object_parameterization():
    """
    few-shot learn symbolic latent parameters to try to fit a real-world object
    """

    datadir = "./data/static_scan"
    images = fetch_images(datadir)
    recap = False
    if images is None or recap:
        scanner = RSCamera(datadir)
        scanner.save_scan(5, None, burnin = 200)
        images = fetch_images(datadir)

    b.setup_visualizer()
    b.clear()

    rgbd = images[4]
    rgbd.depth = rgbd.depth.astype(np.float64) * 34.0 / 65536

    # subsample dense points
    scaling_factor = 0.19
    rgbd_scaled_down = b.RGBD.scale_rgbd(rgbd, scaling_factor)
    pts = b.unproject_depth(rgbd_scaled_down.depth, rgbd_scaled_down.intrinsics)

    # infer grounding plane using RANSAC
    table_pose, plane_dims = b.utils.infer_table_plane(
        pts, jnp.eye(4),
        rgbd_scaled_down.intrinsics,
        ransac_threshold=0.001, inlier_threshold=0.001, segmentation_threshold=0.1
    )

    # visualize
    ## depth cloud
    b.show_cloud("0", pts.reshape(-1,3))
    ## grounding plane
    b.show_pose("table", table_pose)

    b.setup_renderer(rgbd_scaled_down.intrinsics)
    # b.RENDERER.add_mesh_from_file(b.utils.get_assets_dir() + "/sample_objs/toy_plane.ply")
    model_dir = os.path.join(b.utils.get_assets_dir(),"bop/ycbv/models")
    mesh_path = os.path.join(model_dir,"obj_" + "{}".format(13+1).rjust(6, '0') + ".ply")

    # try relax `scale_f` with auxiliary `Normal` randomness.
    mean_scale = 1.0/1600.0
    stdev_scale = 1.0/16000.0
    scale_f = Normal.random(mean_scale, stdev_scale)
    print("logpdf of scale =", Normal.logpdf(scale_f, mean_scale, stdev_scale))
    b.RENDERER.add_mesh_from_file(mesh_path, scaling_factor=scale_f)

    # jitted data-driven proposal
    cpe = ContactParamEnumerator(table_pose)
    obs_img = b.unproject_depth_jit(rgbd_scaled_down.depth, rgbd_scaled_down.intrinsics)
    best_cps, best_indices = cpe.argmax(obs_img)
    poses = cpe.cps_to_pose(best_cps, best_indices)

    # TODO: fit mesh parameters

    # for now we only have one mesh
    b.show_trimesh(f"mesh_0", b.RENDERER.meshes[0])
    b.set_pose(f"mesh_0", poses[0])

    while True:
        cv2.waitKey(1)


if __name__ == "__main__":
    params = learn_object_parameterization()
