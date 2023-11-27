import numpy as np
import sys
sys.path.append("../")
from CAD_transf import get_points
from scipy.spatial.transform import Rotation as R

def create_perspective_matrix(fov_deg, aspect_ratio, z_near, z_far):
    fov_rad = np.radians(fov_deg)
    f = 1.0 / np.tan(fov_rad / 2.0)
    
    projection_matrix = np.array([
        [f / aspect_ratio, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (z_far + z_near) / (z_near - z_far), (2 * z_far * z_near) / (z_near - z_far)],
        [0, 0, -1, 0]
    ])

    return projection_matrix

def threeD_points(cam_pos, cam_rot):
    cam_pos = np.array(cam_pos)
    points = get_points()
    
    offset = points[0][2]
    points = np.array(points)
    points[:, 2] -= offset
    points = np.array(points)*100

    cam_rot = np.array([np.radians(angle) for angle in cam_rot])
    r_mtx = R.from_euler('xyz', cam_rot).as_matrix()
    r_mtxT = np.transpose(r_mtx, (0, 2, 1))

    t_vec = -np.matmul(r_mtxT, cam_pos[:, :, np.newaxis]).squeeze()
    t_vec = np.array([np.append(p,1) for p in t_vec])
    
    t_mtx = np.zeros((r_mtxT.shape[0], 4, 3))
    t_mtx[:, :3, :3] = r_mtxT
    t_mtx = np.concatenate((t_mtx, t_vec[:, :, np.newaxis]), axis=2)

    points = np.array([np.append(p,1) for p in points])
    points_cam = np.matmul(t_mtx, points.T)
    points_cam = points_cam[:, :3, :].squeeze().transpose(0,2,1)
    
    return points_cam

def twoD_points(cam_pos, cam_rot, image_width=512, image_height=512):
    points3D = threeD_points(cam_pos, cam_rot)
    fov_degrees = 67.8
    aspect_ratio = 1
    z_near = 0.2
    z_far = 2.5
    projection_matrix = create_perspective_matrix(fov_degrees, aspect_ratio, z_near, z_far)

    points3D = np.concatenate([points3D, np.ones((points3D.shape[0], points3D.shape[1], 1))], axis=-1)

    points2D = np.matmul(projection_matrix, points3D.transpose(0,2,1)).transpose(0,2,1)
    points2D = points2D / points2D[:, :, 3, np.newaxis]
    points2D = points2D[:, :, :2]
    points2D[:,:,0] = (points2D[:,:,0] + 1) * image_width / 2
    points2D[:,:,1] = (1 - points2D[:,:,1]) * image_height / 2

    visibility_param = np.ones((points2D.shape[0], points2D.shape[1], 1))
    visibility_param[points2D[:,:,0] < 0] = 0
    visibility_param[points2D[:,:,0] > image_width] = 0
    visibility_param[points2D[:,:,1] < 0] = 0
    visibility_param[points2D[:,:,1] > image_height] = 0

    return points2D, visibility_param


