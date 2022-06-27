# PointCloud Preprocessor
## Keyframes Extraction
### mulran_point_cloud_select
Extract keyframes from MulRan dataset using same spatial distance for consecutive keyframes. 
```
# params are specified within the source file
mulran_point_cloud_select
```

### kitti_point_cloud_select
Extract keyframes from KITTI Odometry dataset using same spatial distance for consecutive keyframes. 
```
# params are specified within the source file
kitti_point_cloud_select
```

### oxford_point_cloud_select
Extract keyframes from Oxford Radar RobotCar dataset using same spatial distance for consecutive keyframes. 
```
# params are specified within the source file
oxford_point_cloud_select
```
## Point Cloud Manipulation and BEV Generation
### cloud_manip
Apply a regid transform on a single input pcd pointcloud, and generate BEV image for both original and transformed point clouds.
```
# translation (trans_x, trans_y, trans_z) in metre
# rotation on yaw axis, in degree
cloud_manip <input_pcd> <trans_x> <trans_y> <trans_z> <theta>
```

### batch_cloud_manip
Generate BEV image all point clouds within the specified keyframes folder.
```
batch_cloud_manip <keyframes_root_dir>
```
## Pose Estimation
### top_part_registration
Two-stage pose estimation using flattened 2D point clouds for a single matching pair.
```
# two point clouds to be aligned: query_pcd and match_pcd. 
# initial guess for relative rotation on yaw axis, in degree. 
top_part_registration <query_pcd> <match_pcd> <yaw_initial_guess>
```

### batch_top_part_registration
Perform two-stage pose estimation on a batch of point cloud pairs.
```
# need a match result text file "match_result.txt" descripting all matching pairs.
# each entry contains one matching pair: "query_idx match_idx yaw_initial_guess"
batch_top_part_registration <match_result_text_file> <point_cloud_dir>
```