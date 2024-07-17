import os
import json
import torch
import imageio
import json
import numpy as np
from tqdm import tqdm
from argparse import Namespace
from scene.dataset_readers import readFaceRigSingleFrameInfo
from scene.gaussian_model import GaussianModel
from utils.camera_utils import loadCam
from gaussian_renderer import render

"""
This script will render the gaussian splatting mask of `gaussian_splatting_path` 
using the cameras in `dataset_path` and save the images in `output_path`.
It will change the mask path name in the `output_path/vps05_camera_gsbuffer.json`.
"""
dataset_path = "/root/public/gss/VPS05_kevin_diffuse_frm38/vps05_camera.json"
gaussian_splatting_path = "/root/public/gss/results/kevin_frm38/point_cloud/iteration_30000/point_cloud.ply"
output_path = "/root/public/gss/VPS05_kevin_relit02_0_1_frm38/"
bg_color = torch.zeros((3), device="cuda")
args = Namespace(
    resolution = 1.,
    data_device = "cuda"
)
pipe_args = Namespace(
    convert_SHs_python = False,
    compute_cov3D_python = False,
    debug = False,
)

scene_info = readFaceRigSingleFrameInfo(dataset_path, 1.0, eval=False)

camera_list = []

for id, c in enumerate(scene_info.train_cameras):
    camera_list.append(loadCam(args, id, c, 1.0))

gaussians = GaussianModel(sh_degree=3)
gaussians.load_ply(gaussian_splatting_path)

for camera in tqdm(camera_list):
    buffer_color = torch.ones_like(gaussians.get_xyz)

    # render depth at channel 0
    xyz = gaussians.get_xyz
    xyzw = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=-1)
    view_mat = camera.world_view_transform
    xyzw_view = xyzw @ view_mat[:, :3]
    zbuffer = xyzw_view[:, 2:3]
    zbuffer = torch.nan_to_num(zbuffer, 0, 0, 0)
    zbuffer_max = zbuffer.max()
    zbuffer = zbuffer / zbuffer_max
    buffer_color[:, 0] = zbuffer[:, 0]
    
    buffer_pkg = render(camera, gaussians, pipe_args, 
                        bg_color=torch.tensor([0., 0., 0.]).type_as(zbuffer), 
                        override_color=buffer_color)
    
    depth_map = buffer_pkg["render"][0] * zbuffer_max
    depth_map = depth_map.detach().cpu().numpy()

    out_img_name = camera.image_name.split(".")[0] + "_gsdepth.exr"
    output_file = os.path.join(output_path, "images", out_img_name)
    imageio.imwrite(output_file, depth_map)

    # render mask
    alpha = buffer_pkg["render"][1]
    alpha = alpha.detach().clamp(0, 1.).cpu().numpy()
    
    out_img_name = camera.image_name.split(".")[0] + "_gsmask.png"
    output_file = os.path.join(output_path, "images", out_img_name)
    imageio.imwrite(output_file, (alpha * 255).astype(np.uint8))

# change the mask file name in the camera json
jsonfile = os.path.join(output_path, "vps05_camera.json")
with open(jsonfile, "r") as f:
    data = json.load(f)
    for cam_data in data:
        mask_path = cam_data["image_path"].split(".")[0] + "_gsmask.png"
        if not os.path.exists(os.path.join(output_path, "images", mask_path)):
            print(f"Mask file {mask_path} not found!")
        cam_data["mask_path"] = mask_path

        depth_path = cam_data["image_path"].split(".")[0] + "_gsdepth.exr"
        if not os.path.exists(os.path.join(output_path, "images", depth_path)):
            print(f"Mask file {depth_path} not found!")
        cam_data["depth_path"] = depth_path

jsonfile = os.path.join(output_path, "vps05_camera_gsbuffer.json")
with open(jsonfile, "w") as f:
    json.dump(data, f, indent=4)
