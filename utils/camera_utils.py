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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
from typing import Optional, Union
import torch

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    resized_image_rgb = PILtoTorch(cam_info.image)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    bg_image = PILtoTorch(cam_info.bg_image) if cam_info.bg_image is not None else None
    loaded_mask = PILtoTorch(cam_info.mask) if cam_info.mask is not None else None
    depth_map = torch.tensor(cam_info.depth) if cam_info.depth is not None else None

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, cx=cam_info.cx, cy=cam_info.cy,
                  image=gt_image, gt_alpha_mask=loaded_mask, bg_image=bg_image, depth_map=depth_map,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry


def _save_camera_mesh(    
    path: str,
    extrinsic: np.ndarray,
    intrinsic: Optional[np.ndarray] = None,
    focal: Optional[np.ndarray] = None,
    cxcy: Optional[np.ndarray] = None,
    HW: Optional[np.ndarray] = None,
    isc2w: bool = True,
    camera_size: Union[str, float] = "auto",
    near_far: Optional[np.ndarray] = None,
    y_down: bool = True, 
    z_forward: bool = True,
    x_right: bool = True,
):
    """
    Args:
        path: path of the trimesh
        extrinsic:  R | T    or     -R^T | -R^T C, can be shape of [3/4, 4], [B, 3/4, 4]

        intrinsic:  can be shape of [B, 3, 3] or [3, 3]
        focal: can be shape of [B, 2] or [B] or [1,], indicating [fx, fy] or focal length
        cxcy: shape of [B, 2] or [2,]
            Warning! note that when intrinsic is specified, focal and cxcy will not be used
        HW: image resolution. If not specified, will use cx * 2, cy * 2

        isc2w: when true, extrnsic is the matrix that transform camera coordinate to world coordinate (R | T)
        camera_size: when "auto", size=minimum distance of camera / 2, when float, size=camera_size
        near_far: when not None, should be size of [B, 2], which means (near, far)
    Returns:
        None, will save to path
    """
    default_focal = 1000
    default_cxcy = [500, 500]

    # standardlizing the matrices
    extrinsic = np.array(extrinsic).astype(np.float32)
    if extrinsic.ndim == 2:
        extrinsic = extrinsic[None, ...]
    assert (
        extrinsic.ndim == 3
    ), f"extrinsic should be of shape [B, 3/4, 4], but got {extrinsic.shape}"
    ncamera = len(extrinsic)
    if extrinsic.shape[1] == 3:
        extrinsic = np.concatenate(
            [extrinsic, np.zeros_like(extrinsic[:, :1, :])], axis=1
        )
        extrinsic[:, -1, -1] = 1.0
    assert extrinsic.shape[1:] == (
        4,
        4,
    ), f"extrinsic should be of shape [B, 3/4, 4], but got {extrinsic.shape}"

    if intrinsic is None:  # use focal, cxcy
        if focal is None:
            focal = default_focal
        if cxcy is None:
            cxcy = default_cxcy
        
        focal = np.array(focal).astype(np.float32)
        if focal.ndim == 0:
            focal = focal[None]
        if focal.ndim == 1:
            focal = focal[None]
        assert (focal.ndim == 2), f"focal should be of shape [B, 1/2] or [1/2,]"
        focal = np.broadcast_to(focal, (ncamera, 2))
        cxcy = np.array(cxcy).astype(np.float32)
        if cxcy is None:
            cxcy = cxcy[None]
        if cxcy.ndim == 1:
            cxcy = cxcy[None]
        assert (cxcy.ndim == 2), f"cxcy should be of shape [B, 2] or [2,]"
        cxcy = np.broadcast_to(cxcy, [ncamera, 2])
        intrinsic = np.broadcast_to(np.eye(3)[None], [ncamera, 3, 3]).copy()
        intrinsic[:, [0, 1], [0, 1]] = focal
        intrinsic[:, :2, 2] = cxcy
    else:  # use intrinsic
        intrinsic = np.array(intrinsic).astype(np.float32)
        if intrinsic.ndim == 2:
            intrinsic = intrinsic[None, ...]
        assert (
            intrinsic.ndim == 3
        ), f"intrinsic should be of shape [B, 3, 3], but got {intrinsic.shape}"
        intrinsic = np.broadcast_to(intrinsic, (ncamera, 3, 3))

    if HW is None:
        HW = intrinsic[:, [1, 0], 2] * 2
    else:
        HW = np.array(HW).astype(np.float32)
        if HW.ndim == 1:
            HW = HW[None, ...]
        assert HW.ndim == 2, f"HW should be of shape [B, 2] or [2,]"
        HW = np.broadcast_to(HW, [ncamera, 2])

    # inverse the extrinsic
    if not isc2w:
        poses = np.linalg.inv(extrinsic)
    else:
        poses = extrinsic

    # figure out the camera scale
    if camera_size == "auto":
        camera_pos = poses[:, :3, 3]
        distance = camera_pos[:, None, :] - camera_pos[None, :, :]
        distance = np.linalg.norm(distance, axis=-1)
        distance = distance[distance > 0]
        if len(distance) > 0:
            distance = np.sort(distance)
            camera_size = distance[len(distance) // 100] * 0.5
        else:
            camera_size = 1
        print(f"camera_size = {camera_size}")

    camera_size = float(camera_size)
    assert isinstance(camera_size, float), "camera_size should be auto or float"
    
    # figure out the camera color
    color = np.zeros((ncamera, 3)).astype(np.float32)
    if ncamera < 25:
        color[:, 0] = np.arange(ncamera) * 10 / 255
        color[:, 1:3] = 20 / 255
    elif ncamera < 125:
        color[:, 0] = np.arange(ncamera) * 2 / 255
        color[:, 1:3] = 20 / 255
    elif ncamera < 255:
        color[:, 0] = np.arange(ncamera) / 255
        color[:, 1:3] = 20 / 255
    elif ncamera < 25 * 255:
        color[:, 0] = (np.arange(ncamera) % 255) // 255 / 255
        color[:, 1] = (np.arange(ncamera) // 255) * 10 / 255
        color[:, 2] = 20 / 255
    else:
        color[:, 0] = (np.arange(ncamera) % 255 % 255) / 255
        color[:, 1] = (np.arange(ncamera) % 255 // 255) / 255
        color[:, 2] = (np.arange(ncamera) // 255 // 255) / 255

    # the canonical camera
    camera_faces = [
        [0, 2, 1],
        [0, 4, 2],
        [0, 3, 4],
        [0, 1, 3],
        [5, 6, 7],
        [8, 9, 10],
    ]
    if near_far is not None:
        assert len(near_far) == len(
            extrinsic
        ), "near_far should have save len as extrinsic"
        camera_faces += [[11, 12, 13], [14, 15, 16]]
    camera_faces = np.array(camera_faces).astype(np.int32)

    all_vertices = []
    vertices_count = 0
    all_faces = []
    all_color = []

    for idx, (intrin_, pose_, hw_) in enumerate(zip(intrinsic, poses, HW)):
        h, w = hw_
        intr22inv = np.linalg.inv(intrin_[:2, :2])
        tl = (intr22inv @ -intrin_[:2, 2:3]).reshape(-1)
        tr = (intr22inv @ (np.array([[w, ], [0, ],]) - intrin_[:2, 2:3])).reshape(-1)
        bl = (intr22inv @ (np.array([[0, ], [h, ],]) - intrin_[:2, 2:3])).reshape(-1)
        br = (intr22inv @ (np.array([[w, ], [h, ],]) - intrin_[:2, 2:3])).reshape(-1)
        camera_vertices = [
            [0.0, 0.0, 0.0],
            [tl[0], tl[1], 1],  # tl
            [tr[0], tr[1], 1],  # tr
            [bl[0], bl[1], 1],  # bl
            [br[0], br[1], 1],  # br
            # tops
            [tl[0] * 0.8, tl[1] * 1.1, 1],
            [tr[0] * 0.8, tr[1] * 1.1, 1],
            [0.0, tl[1] * 1.5, 1],
            # right
            [tr[0] * 1.01, tr[1] * 0.1, 1],
            [br[0] * 1.01, br[1] * 0.1, 1],
            [tr[0] * 1.4, 0, 1],
        ]
        if near_far is not None:
            near, far = near_far[idx] / camera_size
            camera_vertices += [
                [-0.5, -0.5, near],
                [0.5, -0.5, near],
                [0, 0.5, near],
                [-0.5, -0.5, far],
                [0.5, -0.5, far],
                [0, 0.5, far],
            ]

        camera_vertices = np.array(camera_vertices).astype(np.float32) * camera_size
        y = np.array([0, -1, 0]) if y_down else np.array([0, 1, 0])
        z = np.array([0, 0, 1]) if z_forward else np.array([0, 0, -1])
        x = np.cross(y, z)
        camera_vertices[:, 0] *= (1 if x_right else -1)
        camera_vertices[:, 1] *= (1 if y_down else -1)
        camera_vertices[:, 2] *= (1 if z_forward else -1)
        camera_vertices = np.concatenate(
            [camera_vertices, np.ones_like(camera_vertices[:, :1])], axis=-1
        )
        camera_vertices = pose_[None, ...] @ camera_vertices[..., None]
        camera_vertices = camera_vertices[:, :3, 0] / camera_vertices[:, 3:, 0]

        camera_color = np.ones_like(camera_vertices).astype(np.float32)
        camera_color[:] = color[idx]

        all_vertices.append(camera_vertices.copy())
        all_color.append(camera_color.copy())
        all_faces.append(camera_faces + vertices_count)
        vertices_count += len(camera_vertices)

    all_vertices = np.concatenate(all_vertices, axis=0)
    all_color = np.concatenate(all_color, axis=0)
    all_faces = np.concatenate(all_faces, axis=0)
    _save_obj_with_vcolor(path, all_vertices, all_color, all_faces)


def _save_obj_with_vcolor(file, verts, colors, faces):
    with open(file, "w") as f:
        for pos, color in zip(verts, colors):
            f.write(f"v {pos[0]} {pos[1]} {pos[2]} {color[0]} {color[1]} {color[2]}\n")

        faces1 = faces + 1
        for face in faces1:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")

        f.write("\n")
