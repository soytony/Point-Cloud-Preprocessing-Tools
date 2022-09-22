# Point Cloud Preprocessing Tools

## 0. Compiling

You may need `OpenCV`, `pcl`, `Eigen3` installed before compiling. 

```
mkdir build
cd build
cmake ..
make -j8
cd ..
```



## 1. Keyframes Extractors

These binaries select keyframes based on the interval distance between two consecutive frames. 
You may specify the interval manually, e.g. `1` or `2`, in meters. 
After running the binaries, you will have files organized as follows: 

```
[keyframes_root_dir]
├ ... 
└ selected_keyframes_xxm/ 
  ├ keyframe_point_cloud/ <- folder for keyframe point clouds in pcd format
  ├ keyframe_pose.csv <- gt poses for all selected keyframes
  └ keyframe_pose_format.csv <- format for one entry in the gt poses
```



### 1.1 mulran_point_cloud_select
Extract keyframes from MulRan dataset using same spatial distance for consecutive keyframes. 
```
Usage:
./build/mulran_point_cloud_select [dataset_root_dir] [keyframe_dist_interval](default=2)

[dataset_root_dir] should be organized as follows:
[dataset_root_dir]
├ sensor_data
│ ├ Ouster/
│ └ ouster_front_stamp.csv
└ global_pose.csv
```

### 1.2 kitti_point_cloud_select
Extract keyframes from KITTI Odometry dataset using same spatial distance for consecutive keyframes. 
```
Usage: 
./build/kitti_point_cloud_select [dataset_root_dir] [keyframe_dist_interval](default=2)

[dataset_root_dir] should be organized as follows:
[dataset_root_dir]
├ velodyne/
├ times.txt
└ global_pose.txt
```

### 1.3 oxford_point_cloud_select
Extract keyframes from Oxford Radar RobotCar dataset using same spatial distance for consecutive keyframes. 
```
Usage: 
./build/oxford_point_cloud_select [dataset_root_dir] [keyframe_dist_interval](default=2)

[dataset_root_dir] should be organized as follows:
[dataset_root_dir]
├ velodyne_left/
├ velodyne_left.timestamps
└ gps/
  └ ins.csv
```
## 2. Point Cloud Manipulator
### 2.1 cloud_manip
Apply a regid transform on a single input pcd pointcloud, and generate BEV image for both original and transformed point clouds.
```
Usage:

# translation (trans_x, trans_y, trans_z) in metre
# rotation on yaw axis, in degree

./build/cloud_manip [input_pcd] [trans_x] [trans_y] [trans_z] [theta]
```

### 2.2 batch_cloud_manip
Generate BEV image all point clouds within the specified keyframes folder.
```
Usage:
./build/batch_cloud_manip [keyframes_root_dir]
```


## 3. BEV Generator

This binary generates ground-removed point clouds, single & multi layer BEV images and creates geometric distance-based labels for each point cloud. 

After running the binary, you will have files organized as follows: 

```
[keyframes_root_dir]
├ ...
├ non_ground_point_cloud/ <- folder for ground-removes point clouds in pcd format
├ output_multi_bev/ <- folder for multi-layer BEV images
└ output_single_bev/ <- folder for single-layer BEV images
```

Single-layer BEV images are provided in `csv` and `png` format, and multi-layer BEV images are provided in raw binaries `bin` and `png` format. 

```
Usage: 
./build/batch_multi_bev_gen [keyframes_root_dir] [sensor_type]

[keyframes_root_dir] should be organized as follows: 
[keyframes_root_dir]
├ keyframe_point_cloud/ <- folder for selected point clouds in pcd format for each frame
├ keyframe_pose.csv <- 6-DoF pose for each frame
└ keyframe_pose_format.csv <- 6-DoF pose format description

[sensor_type] could be HDL_32E, HDL_64E or OS1_64. 
```



## 4. Relative Pose Estimator

### 4.1 top_part_registration
Two-stage pose estimation using flattened 2D point clouds for a single matching pair.
```
Usage:

# two point clouds to be aligned: query_pcd and match_pcd. 
# initial guess for relative rotation on yaw axis, in degree. 
./build/top_part_registration [query_pcd] [match_pcd] [yaw_initial_guess]
```

### 4.2 batch_top_part_registration
Perform two-stage pose estimation on a batch of point cloud pairs.
```
Usage:

# need a match result text file "match_result.txt" descripting all matching pairs.
# each entry contains one matching pair: "query_idx match_idx yaw_initial_guess"
./build/batch_top_part_registration [match_result_text_file] [point_cloud_dir]
```



## 5. Data Preprocessing for FreSCo

### Step 1:

Use `Keyframes Extractors` to extract keyframes from your datasets. 3 different extractors are provided to handle different datasets. 

### Step 2:

Use `BEV Generator` to convert the `pcd`-format keyframes to BEV images. The `csv`-format single-layer BEV images and their corresponding ground truth poses will be used in FreSCo for place retrieval and performance estimation. 

