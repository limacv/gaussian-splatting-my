import numpy as np
import json
import os
import cv2
import torch
from tqdm import tqdm
import imageio
from utils.graphics_utils import focal2fov, fov2focal
from scene.dataset_readers import storePly
from utils.camera_utils import _save_camera_mesh


output_path = "/root/public/gss/VPS05_kevin_diffuse_frm38"

frm_data_json_path = "/fsx_scanline/from_eyeline/FaceRig/structured_data/vps05_kevin_2023112901/gs_4k/data/dynamic_0009/frame_info.json"
cam_data_json_path = "/fsx_scanline/from_eyeline/FaceRig/structured_data/vps05_2023/vps05_kevin_2023112901/gs_4k/camera/camera_info.json"
pts_data_npz_path = "/fsx_scanline/from_eyeline/FaceRig/structured_data/vps05_kevin_2023112901/gs_4k/data/dynamic_0009/init_pt_cld.npz"
bg_frm_data_path = "/fsx_scanline/from_eyeline/FaceRig/vpintel/vpintel2023112901_KW0010/raw/data/dynamic_0003/frame_0135"
frm_start = 38
frm_end = 38
frm_interval = 1
image_downscale = 3 # 1280 x 720
image_convert_linear_to_srgb = True
image_clip_to_0_1 = True
image_color_scale = 1.5
image_color_gamma = 2.4
linear_color_clamp = 2.5
enable_undistort = False
crop_bbox_min_x = 0.
crop_bbox_min_y = 0.
crop_bbox_min_z = 0.
crop_bbox_max_x = 0.
crop_bbox_max_y = 0.
crop_bbox_max_z = 0.
edge_width_undistort = 40
eval_cam_ids = []  # [1, 15, 30, 35, 45, 60, 75]


image_save_path = os.path.join(output_path, "images")
os.makedirs(image_save_path, exist_ok=True)
camera_save_file = os.path.join(output_path, "vps05_camera.json")
camera_visual_file = os.path.join(output_path, "camera_vis.obj")
camera_infos = []


def _load_image(img_filepath, im_height, im_width, convert_linear_to_srgb, clip_to_0_1, down_scale, gamma, linear_color_clamp, 
            dynamic_intensity_scale = 1., cam_in = None, cam_coef = None, msk = None, img_to_merge = None, merge_scale0 = 1., merge_scale1 = 1.):
    """
    load image and turn it into torch tensor with shape [C,H,W]
    """
    def linear_to_scaled_srgb(img, down_scale, gamma):
        limit = 0.0031308
        img = torch.where(img > limit, 1.055 * img ** (1 / gamma) - 0.055, 12.92 * img)
        img = img * (1.0 / down_scale)
        return img


    def linear_to_scaled_srgb_with_clamp(img, gamma, clamp):
        limit = 0.0031308
        img = torch.where(img > limit, 1.055 * img ** (1 / gamma) - 0.055, 12.92 * img)
        thresh = 1.055 * clamp ** (1 / gamma)
        img = img * (1.0 / thresh)
        return img

    orig_img = cv2.imread(img_filepath, cv2.IMREAD_UNCHANGED)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

    # HMM: to do undistortion
    if cam_in is not None and cam_coef is not None:
        orig_img = cv2.undistort(orig_img, cam_in, cam_coef)
    
    # HMM: resize tensor if size does not match
    h, w, _ = orig_img.shape
    if h != im_height or w != im_width:
        orig_img = cv2.resize(orig_img, (im_width, im_height))

    img = torch.from_numpy(orig_img).cuda()
    img = img.permute(2, 0, 1)  # [H,W,3] -> [3,H,W]
    img = img.to(torch.float16) * dynamic_intensity_scale

    if msk is not None:
        print("HMM: img shape = ", img.shape, ", msk shape = ", msk.shape)
        img = img * msk + (1. - msk)

    if img_to_merge is not None:
        img = img * merge_scale0 + img_to_merge * merge_scale1
    
    if img.dtype == torch.uint8:
        img = img / 255.
        # HMM: if there exists undistorted image mask
    else:
        # HMM: if there exists undistorted image mask
        if linear_color_clamp > 0.:
            img = linear_to_scaled_srgb_with_clamp(img, gamma, linear_color_clamp)
        elif convert_linear_to_srgb:
            # what we get here is not srgb with gamma compression
            # and that gamma compression seems to be better for learning
            # we should account for when we may get linear space color images
            #img = linear_to_srgb(img)
            img = linear_to_scaled_srgb(img, down_scale, gamma)
    if clip_to_0_1:
        img = torch.clamp(img, 0, 1)
    
    return img

###########################################################################################################
# Actual reading
###########################################################################################################

md = json.load(open(frm_data_json_path, 'r'))  # metadata
cd = json.load(open(cam_data_json_path, "r"))  # HMM: camera data

cam_bk_msks = {} # HMM: load background mask for each camera if there exists
cam_infos_unsorted = []
frm_num = int((frm_end - frm_start) / frm_interval) + 1

rotation_left = [1, 2, 3, 4, 9, 10, 11, 12, 18, 19, 20, 21, 22, 28, 29, 30, 31, 32, 38, 39, 40, 41, 42, 48, 49, 50, 51, 52, 57, 62, 63, 64, 65, 70, 71, 72, 73]

for f in range(frm_start, frm_end + 1, frm_interval):
    frm_data = md['frames'][str(f)]  
        
    if frm_num <= 1:
        frm_time = 0.
    else:
            # frm_time = (f - frm_start) / (frm_end - frm_start)
            # HMM: decide frame time based on the larger time range
            range_start = min(frm_start, range_start)
            range_end   = max(frm_end, range_end)
            frm_time = (f - range_start) / float(range_end - range_start)

    print("HMM: ######## f = ", f, frm_time)

    # c = 0
    Rs, Ts = [], []
    for fn in tqdm(frm_data.keys()):
        cid = frm_data[fn]["camera_index"]
        if str(cid) not in cd.keys():
            continue
        # c += 1
        # if c >= 20:
        #     break
        cam_data = cd[str(cid)]

        cam_R = cam_data["R"]
        cam_T = cam_data["T"]
        cam_in = np.array(cam_data["intrinsics"])
        cam_w = cam_data["width"]
        cam_h = cam_data["height"]
        cam_coef = None
        if enable_undistort:
            k1 = cam_data["lens_coefficients_k1"]
            k2 = cam_data["lens_coefficients_k2"]
            p1 = cam_data["lens_coefficients_p1"]
            p2 = cam_data["lens_coefficients_p2"]
            k3 = cam_data["lens_coefficients_k3"]
            cam_coef = np.array((k1, k2, p1, p2, k3))
            
        R = np.transpose(cam_R)
        T = np.array(cam_T)
        focal_length_x = cam_in[0, 0]
        focal_length_y = cam_in[1, 1]
        if abs(focal_length_x) < 1e-5:
            continue
        
        FovY = focal2fov(focal_length_x, cam_h)
        FovX = focal2fov(focal_length_y, cam_w)

        # to be corrected in json file itself! ( Fix /fsx_vfx path here for now! )
        # HMM@HACK
        # im_path = fn.replace('/fsx_vfx','/fsx_scanline')
        # im_path = fn.replace('/fsx_scanline/from_eyeline','/mnt/HMM/Data')
        im_path = fn
        im_name = os.path.basename(im_path)
        im = None
        # im_width = w
        # im_height = h
        down_width = int(cam_w / float(image_downscale))
        down_height = int(cam_h / float(image_downscale))
        bg_im = None
        diff_im = None
        
        bg_image_filepath = os.path.join(bg_frm_data_path, fn.split('/')[-1])
        
        if os.path.exists(bg_image_filepath):
            bg_im = _load_image(bg_image_filepath, down_height, down_width,
                                True,
                                image_clip_to_0_1,
                                image_color_scale, 
                                image_color_gamma,
                                linear_color_clamp,
                                1.,
                                cam_in, cam_coef, None, bg_im.transpose(2, 0, 1) if bg_im is not None else None, 0.1, 1.)
            bg_im = bg_im.to(dtype=torch.float16)
            bg_im = bg_im.cpu().numpy().transpose(1, 2, 0)
            # print("HMM: bg_image_filepath = ", bg_image_filepath, bg_im.shape)
            
        im = _load_image(im_path, down_height, down_width,
                            image_convert_linear_to_srgb, 
                            image_clip_to_0_1,
                            image_color_scale, 
                            image_color_gamma,
                            linear_color_clamp,
                            1.,
                            cam_in, cam_coef, cam_bk_msks[cid] if cid in cam_bk_msks.keys() else None, diff_im, 1., 0.1)
        
        im = im.cpu().numpy().transpose(1, 2, 0)

        # apply rotation
        extrin_r = np.array(cam_data["R"])
        fx, fy = cam_in[0, 0], cam_in[1, 1]
        cx, cy = cam_in[0, 2], cam_in[1, 2]
        extrin = np.concatenate([extrin_r, T[..., None]], axis=-1)
        camera_right = extrin_r.T[:, 0]
        if camera_right[1] < 0:  # image rotate right
            im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
            bg_im = cv2.rotate(bg_im, cv2.ROTATE_90_CLOCKWISE) if bg_im is not None else None
            extrin = np.stack([-extrin[1], extrin[0], extrin[2]])
            intrin = np.array([[fy, 0, cam_h - cy], [0, fx, cx], [0, 0, 1]])
        else:  # camera_right[1] < 0  # image rotate left
            im = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
            bg_im = cv2.rotate(bg_im, cv2.ROTATE_90_COUNTERCLOCKWISE) if bg_im is not None else None
            extrin = np.stack([extrin[1], -extrin[0], extrin[2]])
            intrin = np.array([[fy, 0, cy], [0, fx, cam_w - cx], [0, 0, 1]])
        
        cam_data["R"] = extrin[:3, :3].tolist()
        cam_data["T"] = extrin[:3, 3].tolist()
        cam_data["intrinsics"] = intrin.tolist()
        cam_data["width"], cam_data["height"] = cam_h, cam_w

        save_name = im_name.split('.')[0] + ".png"
        imageio.imwrite(os.path.join(image_save_path, save_name), (im * 255).astype(np.uint8))
        cam_data["image_path"] = save_name
        if bg_im is not None:
            imageio.imwrite(os.path.join(image_save_path, save_name.split('.')[0] + "_bg.png"), (bg_im * 255).astype(np.uint8))
            cam_data["bg_path"] = save_name.split('.')[0] + "_bg.png"
        
        camera_infos.append(cam_data)

# save and visualize camera
with open(camera_save_file, 'w') as f:
    json.dump(camera_infos, f)

Rs = np.stack([np.array(info["R"]) for info in camera_infos])
Ts = np.stack([np.array(info["T"]) for info in camera_infos])
extrin = np.concatenate([Rs, Ts[..., None]], axis=-1)
intrin = np.stack([np.array(info["intrinsics"]) for info in camera_infos])
_save_camera_mesh(camera_visual_file, extrin, intrin, isc2w=False)

# Read and save point cloud
# ----------------------------------
ply_path = os.path.join(output_path, "points3D.ply")
#if not os.path.exists(ply_path):
print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
init_pt_cld = np.load(pts_data_npz_path)["data"]
xyz = init_pt_cld[:, :3]
rgb = init_pt_cld[:, 3:6]

# check user crop bbox and application to points
bx = crop_bbox_max_x - crop_bbox_min_x
by = crop_bbox_max_y - crop_bbox_min_y
bz = crop_bbox_max_z - crop_bbox_min_z
if abs(bx*by*bz) > 0.01:    # check if crop bbox has some volume, before we apply cropping       
    # crop by bbox
    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]
    inside_bbox_indices = (x >= crop_bbox_min_x) & (x <= crop_bbox_max_x) & \
                        (y >= crop_bbox_min_y) & (y <= crop_bbox_max_y) & \
                        (z >= crop_bbox_min_z) & (z <= crop_bbox_max_z)
    xyz = xyz[inside_bbox_indices]
    rgb = rgb[inside_bbox_indices]

storePly(ply_path, xyz, rgb)
