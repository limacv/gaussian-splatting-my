#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import *
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.gaussians = gaussians

        self.train_cameras = {}
        self.test_cameras = {}

        if "VPS05" in args.source_path:
            scene_info = readFaceRigSingleFrameInfo(args.source_path, args.facerig_factor)
        elif os.path.exists(os.path.join(args.source_path, "cameras.json") and os.path.join(args.source_path, "mesh.obj")):
            print("Found cameras.json file, assuming Eyeful dataset!")
            scene_info = readEyefulInfo(args.source_path, args.eyeful_subdir, args.eyeful_force_pinhole, args.eval, args.eyeful_loadcamera)
        elif os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = readColmapSceneInfo(args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = readNerfSyntheticInfo(args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms.json")):
            print("Found transforms.json file, assuming nerfstudio data set!")
            scene_info = readNerfstudioInfo(args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if args.initial_gs_model:
            if not os.path.exists(args.initial_gs_model):
                print(f"Could not find initial model at {args.initial_gs_model}")
                exit(-1)
            self.gaussians.load_ply(args.initial_gs_model)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]