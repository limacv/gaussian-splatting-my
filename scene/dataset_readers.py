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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from typing import Optional
from glob import glob

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    cx: Optional[float] = None  # normalized cx, = cx / w - 0.5
    cy: Optional[float] = None  # normalized cy, = cy / w - 0.5
    distcoeffs: Optional[np.array] = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


# for eyeful dataset
def readCamerasFromEyefulCameras(camjson, imageformat, force_undistort):
    import cv2
    cam_infos = []

    with open(camjson) as json_file:
        contents = json.load(json_file)
        ext = None
        for i, cam_info_raw in enumerate(contents["KRT"]):
            print(f"\r read {i}/{len(contents['KRT'])}", end="")
            cameraId = cam_info_raw["cameraId"]
            sensorId = cam_info_raw["sensorId"]
            height, width = cam_info_raw["height"], cam_info_raw["width"]
            intrin = np.array(cam_info_raw["K"]).T
            distort_mod = cam_info_raw["distortionModel"]
            distort = cam_info_raw["distortion"]
            extrin = np.array(cam_info_raw["T"]).T  # world to camera, opencv

            R = np.transpose(extrin[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = extrin[:3, 3]
            
            fx = intrin[0, 0]
            fy = intrin[1, 1]
            cx = intrin[0, 2]
            cy = intrin[1, 2]
            FovX = focal2fov(fx, width)
            FovY = focal2fov(fy, height)
            cx = cx / width - 0.5
            cy = cy / height - 0.5

            if len(distort) == 8:
                k1, k2, k3, _, _, _, p1, p2 = distort
            else:
                k1, k2, p1, p2, k3 = distort
            
            if ext is None:
                image_path = os.path.join(os.path.dirname(camjson), imageformat, f"{cameraId}.*")
                image_path = glob(image_path)[0]
                ext = image_path.split('.')[-1]
            else:
                image_path = os.path.join(os.path.dirname(camjson), imageformat, f"{cameraId}.{ext}")
            image_name = os.path.basename(image_path)

            if ext.lower() in ["jpg", "jpeg", "png"]:
                image = cv2.imread(image_path)[..., ::-1]
            elif ext.lower() == "exr":
                os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
                img = cv2.imread("apartment/images-2k/17/17_DSC0316.exr", cv2.IMREAD_UNCHANGED)
                # tonemap to sRGB
                linear_part = 12.92 * img
                exp_part = 1.055 * (np.maximum(img, 0.0) ** (1 / 2.4)) - 0.055
                image = np.where(img <= 0.0031308, linear_part, exp_part)
            else:
                raise RuntimeError(f"ext = {ext} unrecognized")

            cam_img_h, cam_img_w = image.shape[:2]
            mask = np.ones((cam_img_h, cam_img_w, 1), np.uint8) * 255
            dist = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
            mtx = intrin.copy()
            mtx[0] *= cam_img_w / width
            mtx[1] *= cam_img_h / height
            undistort = cv2.undistort(image, mtx, dist, None, mtx)
            mask = cv2.undistort(mask, mtx, dist, None, mtx)
            
            if distort_mod == "Fisheye" and force_undistort:
                if i == 0:
                    print("Detect Fisheye Camera" + ", Doing undistortion" if force_undistort else ".")
                map1, map2 = cv2.fisheye.initUndistortRectifyMap(mtx, np.array([0, 0, 0, 0.]), np.eye(3), mtx, (cam_img_w, cam_img_h), cv2.CV_16SC2)
                pinhole = cv2.remap(undistort, map1, map2, interpolation=cv2.INTER_AREA)
                mask = cv2.remap(mask, map1, map2, interpolation=cv2.INTER_NEAREST)
                undistort = pinhole
                
            image = np.concatenate([undistort, mask[..., None]], axis=-1)
            image = Image.fromarray(np.array(image, dtype=np.byte), "RGBA")
            cam_infos.append(CameraInfo(uid=cameraId, R=R, T=T, FovY=FovY, FovX=FovX, image=image, cx=cx, cy=cy, 
                                        image_path=image_path, image_name=image_name, width=cam_img_w, height=cam_img_h,
                                        distcoeffs=dist))
        
        print("\n")
        sys.stdout.write('\n')

    return cam_infos


def readEyefulInfo(path, subdir, force_pinhole, eval):
    import trimesh
    
    print("Reading Cameras.json")
    cam_infos = readCamerasFromEyefulCameras(os.path.join(path, "cameras.json"), subdir, force_pinhole)
    
    # read split.json
    with open(os.path.join(path, "splits.json")) as json_file:
        contents = json.load(json_file)
        trains = contents["train"]
        tests = contents["test"]
        train_cam_infos = [caminfo for caminfo in cam_infos if caminfo.uid in trains]
        test_cam_infos = [caminfo for caminfo in cam_infos if caminfo.uid in tests]

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    obj_path = os.path.join(path, "mesh.obj")
    assert os.path.exists(obj_path), "Missing mesh.obj"
    mesh = trimesh.load(obj_path)

    # sample on the mesh
    num_pts = 500_000
    xyz, _ = trimesh.sample.sample_surface_even(mesh, num_pts, seed=0)
    # sample additional points on far-field
    sphere_pt = np.random.randn(20_000, 3).astype(np.float32)
    sphere_pt = sphere_pt / (np.linalg.norm(sphere_pt, axis=-1, keepdims=True) + 0.00001) * 50

    xyz = np.concatenate([xyz, sphere_pt])
    color = np.ones_like(xyz) * 0.1
    normals = xyz / np.linalg.norm(xyz, axis=-1, keepdims=True)
    pcd = BasicPointCloud(points=xyz, colors=color, normals=normals)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=obj_path)
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Eyeful": readEyefulInfo,
}