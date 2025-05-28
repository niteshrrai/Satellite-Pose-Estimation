import numpy as np
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import random


def quat2dcm(q):
    """ Computing direction cosine matrix from quaternion, adapted from PyNav. 
    Arguments:
        q: (4,) numpy.ndarray - unit quaternion (scalar-first)
    Returns:
        dcm: (3,3) numpy.ndarray - corresponding DCM
    """
    # normalizing quaternion
    q = q/np.linalg.norm(q)

    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    dcm = np.zeros((3, 3))

    dcm[0, 0] = 2 * q0 ** 2 - 1 + 2 * q1 ** 2
    dcm[1, 1] = 2 * q0 ** 2 - 1 + 2 * q2 ** 2
    dcm[2, 2] = 2 * q0 ** 2 - 1 + 2 * q3 ** 2
    
    dcm[0, 1] = 2 * q1 * q2 + 2 * q0 * q3
    dcm[0, 2] = 2 * q1 * q3 - 2 * q0 * q2

    dcm[1, 0] = 2 * q1 * q2 - 2 * q0 * q3
    dcm[1, 2] = 2 * q2 * q3 + 2 * q0 * q1

    dcm[2, 0] = 2 * q1 * q3 + 2 * q0 * q2
    dcm[2, 1] = 2 * q2 * q3 - 2 * q0 * q1

    return dcm


def project_keypoints(q_vbs2tango, r_Vo2To_vbs, cameraMatrix, distCoeffs, keypoints):
    ''' Project keypoints.
    Arguments:
        q_vbs2tango:  (4,) numpy.ndarray - unit quaternion from VBS to Tango frame
        r_Vo2To_vbs:  (3,) numpy.ndarray - position vector from VBS to Tango in VBS frame (m)
        cameraMatrix: (3,3) numpy.ndarray - camera intrinsics matrix
        distCoeffs:   (5,) numpy.ndarray - camera distortion coefficients in OpenCV convention
        keypoints:    (3,N) or (N,3) numpy.ndarray - 3D keypoint locations (m)
    Returns:
        points2D: (2,N) numpy.ndarray - projected points (pix)
        points3D: (3,N) numpy.ndarray - 3D points in camera frame
    '''
    if keypoints.shape[0] != 3:
        keypoints = np.transpose(keypoints)

    keypoints_homo = np.vstack((keypoints, np.ones((1, keypoints.shape[1]))))

    R = quat2dcm(q_vbs2tango)
    pose_mat = np.hstack((np.transpose(R), np.expand_dims(r_Vo2To_vbs, 1)))
    xyz = np.dot(pose_mat, keypoints_homo)  # [3 x N]
    
    points3D = xyz
    x0, y0 = xyz[0,:] / xyz[2,:], xyz[1,:] / xyz[2,:]  # [1 x N] each

    # apply distortion
    r2 = x0*x0 + y0*y0
    cdist = 1 + distCoeffs[0]*r2 + distCoeffs[1]*r2*r2 + distCoeffs[4]*r2*r2*r2
    x = x0*cdist + distCoeffs[2]*2*x0*y0 + distCoeffs[3]*(r2 + 2*x0*x0)
    y = y0*cdist + distCoeffs[2]*(r2 + 2*y0*y0) + distCoeffs[3]*2*x0*y0

    # apply camera matrix
    points2D = np.vstack((cameraMatrix[0,0]*x + cameraMatrix[0,2],
                         cameraMatrix[1,1]*y + cameraMatrix[1,2]))

    return points2D/3.0, points3D


def main():
    parser = argparse.ArgumentParser(description='Project 3D keypoints to 2D images')
    parser.add_argument('--src', type=str, required=True, help='Source directory')
    parser.add_argument('--json', type=str, required=True, help='JSON filename')
    parser.add_argument('--images', type=str, required=True, help='Images directory')
    parser.add_argument('--plot', action='store_true', help='Visualize projections')
    args = parser.parse_args()
    
    image_folder_path = Path(args.src) / "images" / args.images
    json_path = Path(args.src) / args.json
    
    camera_data = json.load(open("camera.json"))
    sat_points = np.array(camera_data["sat_points"], dtype=np.float32)
    camera_matrix = np.array(camera_data["cameraMatrix"], dtype=np.float32)
    dist_coeffs = np.array(camera_data["distCoeffs"], dtype=np.float32)
    pose_data = json.load(open(json_path))
    
    plot_count = 0
    
    for entry in pose_data:
        points2D, points3D = project_keypoints(
            np.array(entry['q_vbs2tango_true']),
            np.array(entry['r_Vo2To_vbs_true']),
            camera_matrix, dist_coeffs, sat_points
        )
        
        entry['keypoints_cameraframe'] = points3D.T.tolist()
        entry['keypoints_projected2D'] = points2D.T.tolist()
        
        if args.plot and plot_count < 5:
            try:
                image = Image.open(image_folder_path / entry['filename']).convert('RGB')
                plt.figure(figsize=(10, 8))
                visible_mask = (points2D[1, :] >= 0) & (points2D[1, :] < 1200) & (points2D[0, :] >= 0) & (points2D[0, :] < 1920)
                visible_points = points2D[:, visible_mask]
                plt.imshow(np.array(image))
                plt.scatter(visible_points[0, :], visible_points[1, :], c='r', marker='x', s=40)
                plt.title(entry['filename'])
                plt.show()
                plot_count += 1
            except FileNotFoundError:
                print(f"Warning: {entry['filename']} not found")

    json.dump(pose_data, open(json_path, 'w'), indent=2)
    
    print("Processing complete. JSON file updated.")


if __name__ == "__main__":
    main()


# Example Usage:

# python3 project_keypoints.py --src synthetic --json train.json --images train --plot
# python3 project_keypoints.py --src synthetic --json val.json --images val --plot
# python3 project_keypoints.py --src synthetic --json test.json --images test --plot
# python3 project_keypoints.py --src lightbox --json test.json --images test --plot
# python3 project_keypoints.py --src sunlamp --json test.json --images test --plot


