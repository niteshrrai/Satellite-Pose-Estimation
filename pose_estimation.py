import json
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import argparse

def pnp(points_3D, points_2D, cameraMatrix, distCoeffs=None, rvec=None, tvec=None, useExtrinsicGuess=False):
    '''
    Perform EPnP using OpenCV.
    
    Arguments:
        points_3D: (N,3) numpy.ndarray - 3D coordinates
        points_2D: (N,2) numpy.ndarray - 2D coordinates
        cameraMatrix: (3,3) numpy.ndarray - Camera intrinsic matrix
        distCoeffs: (5,1) numpy.ndarray - Distortion coefficients
        
    Returns:
        q_pr: (4,) numpy.ndarray - unit quaternion (scalar-first)
        t_pr: (3,) numpy.ndarray - position vector (m)
    '''
    if distCoeffs is None:
        distCoeffs = np.zeros((5, 1), dtype=np.float32)
    
    assert points_3D.shape[0] == points_2D.shape[0], 'points 3D and points 2D must have same number of vertices'
    
    points_3D = np.ascontiguousarray(points_3D).reshape((-1, 1, 3))
    points_2D = np.ascontiguousarray(points_2D).reshape((-1, 1, 2))
    
    success, R_exp, t = cv2.solvePnP(points_3D,
                                    points_2D,
                                    cameraMatrix,
                                    distCoeffs, 
                                    rvec, 
                                    tvec, 
                                    useExtrinsicGuess,
                                    flags=cv2.SOLVEPNP_EPNP)
    
    if not success:
        raise Exception("PnP solver failed")
    
    R_pr, _ = cv2.Rodrigues(R_exp)
    
    # Rotation matrix to quaternion
    q_pr = R.from_matrix(R_pr).as_quat()
    
    # Convert from scipy's [x,y,z,w] to scalar-first [w,x,y,z]
    q_pr = np.array([q_pr[3], q_pr[0], q_pr[1], q_pr[2]])
    
    return q_pr, np.squeeze(t)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Pose estimation using PnP algorithm')
    parser.add_argument('--src', type=str, default='subset', 
                        help='Source folder containing test.json file (default: subset)')
    
    args = parser.parse_args()
    
    # Load data
    with open("camera.json", 'r') as f:
        camera_data = json.load(f)
    
    # Construct path to test.json
    test_file_path = f"{args.src}/test.json"
    try:
        with open(test_file_path, 'r') as f:
            test_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Test file '{test_file_path}' not found.")
        return
    
    # Ensure test_data is a list
    if not isinstance(test_data, list):
        test_data = [test_data]
    
    # Extract camera parameters
    camera_matrix = np.array(camera_data["cameraMatrix"], dtype=np.float64)
    dist_coeffs = np.array(camera_data["distCoeffs"], dtype=np.float64)
    
    # Get 3D points (object points)
    sat_points = np.array(camera_data["sat_points"], dtype=np.float64)
    
    # Process all images
    results = []
    total_orientation_error = 0
    total_position_error = 0
    total_pose_score = 0
    
    for i, test_item in enumerate(test_data):
        # Get 2D keypoints (predicted)
        image_points = np.array(test_item["keypoints_projected2D_pred"], dtype=np.float64)
        
        # Ground truth for comparison
        gt_quaternion = np.array(test_item["q_vbs2tango_true"], dtype=np.float64)
        gt_position = np.array(test_item["r_Vo2To_vbs_true"], dtype=np.float64)
        
        # Perform PnP
        try:
            estimated_quaternion, estimated_position = pnp(
                sat_points, 
                image_points, 
                camera_matrix, 
                dist_coeffs
            )

            if np.dot(estimated_quaternion, gt_quaternion) < 0:
                estimated_quaternion = -estimated_quaternion
        
        except Exception as e:
            print(f"Error during PnP estimation for {test_item['filename']}: {e}")
            continue

        
        # Calculate position error
        position_error = np.linalg.norm(gt_position - estimated_position) / np.linalg.norm(gt_position)
        total_position_error += position_error
        
        # Apply machine precision correction for position
        position_score = 0 if position_error < 0.002173 else position_error
        
        # Calculate orientation error
        dot_product = np.abs(np.dot(estimated_quaternion, gt_quaternion))
        dot_product = min(1.0, max(-1.0, dot_product))
        orientation_error_rad = 2.0 * np.arccos(dot_product)
        orientation_error_deg = orientation_error_rad * 180.0 / np.pi
        total_orientation_error += orientation_error_deg
        
        # Apply machine precision correction for orientation
        orientation_score = 0 if orientation_error_deg < 0.169 else orientation_error_deg
        
        # Calculate pose score
        pose_score = orientation_score + position_score
        total_pose_score += pose_score
                
        # Store results
        result = {
            "r_Vo2To_vbs_pred": estimated_position.tolist(),
            "q_vbs2tango_pred": estimated_quaternion.tolist(),
            "Position Error": float(position_error) if isinstance(position_error, np.ndarray) else position_error,
            "Orientation Error": float(orientation_error_deg) if isinstance(orientation_error_deg, np.ndarray) else orientation_error_deg,
            "Pose Score": float(pose_score) if isinstance(pose_score, np.ndarray) else pose_score,
        }
        results.append(result)
        

    # Calculate average scores
    n = len(results)
    avg_pose_score = total_pose_score / n if n else 0
    avg_orientation_error = total_orientation_error / n if n else 0
    avg_position_error = total_position_error / n if n else 0

    
    print(f"\nTotal images processed: {n}")
    print(f"Average Position Error: {avg_position_error:.6f}")
    print(f"Average Orientation Error: {avg_orientation_error:.6f} degrees")
    print(f"Average Pose Score: {avg_pose_score:.6f}")
    

    # Save results to file
    for i, result in enumerate(results):
        test_data[i]["q_vbs2tango_pred"] = result["q_vbs2tango_pred"]
        test_data[i]["r_Vo2To_vbs_pred"] = result["r_Vo2To_vbs_pred"]
        test_data[i]["Position Error"] = result["Position Error"]
        test_data[i]["Orientation Error"] = result["Orientation Error"]
        test_data[i]["Pose Score"] = result["Pose Score"]

    # Save back to test.json
    with open(test_file_path, "w") as f:
        json.dump(test_data, f, indent=2)

    print(f"\nResults appended to {test_file_path}")

if __name__ == "__main__":
    main()

# python pose_estimation.py --src subset
# python pose_estimation.py --src synthetic


