/*
 * @Author: clicheeeeee waterwet@outlook.com
 * @Date: 2022-06-28 10:24:39
 * @LastEditors: clicheeeeee waterwet@outlook.com
 * @LastEditTime: 2022-11-18 18:14:40
 * @FilePath: /pointcloud_preprocessing/BatchMultiBevGen.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
/*
 * @Author: tonyfan waterwet@outlook.com
 * @Date: 2022-04-17 10:24:39
 * @LastEditors: clicheeeeee waterwet@outlook.com
 * @LastEditTime: 2022-09-18 17:24:54
 * @FilePath: /pointcloud_preprocessing/BatchMultiBevGen.cpp
 * @Description: Given a set of key frames and their gt poses, this tool generates multi-layer BEV images and creates smoothed labels for them.
 */
#include "BatchMultiBevGen.h"
#define BACKWARD_HAS_DW 1

// using PointType = PointXYZIRCL;
using PosVecMat = std::vector<std::vector<float>>;
using InvKeyTree = KDTreeVectorOfVectorsAdaptor<PosVecMat, float>;
using LabelType = std::vector<float>; // use one-hot label encoding

const int GROUND_HEIGHT_GRID_ROWS = 75;
const int GROUND_HEIGHT_GRID_COLS = 50;


std::vector<std::pair<int, int>> four_neighbor_iterator_;

std::string output_bvm_dir_;
std::string output_multi_bvm_bin_dir_;
std::string output_multi_bvm_img_dir_;
std::string output_single_bvm_img_dir_;
std::string output_single_bvm_csv_dir_;

SensorParams sensor_params_;

void initDirectories(std::string keyframes_root_dir)
{
    if (keyframes_root_dir.back() != '/') {
        keyframes_root_dir += "/";
    }

    int unused __attribute__((unused));

    //create output bird-view map folder
    output_bvm_dir_ = keyframes_root_dir + "output_multi_bev/";
    unused = system(("rm -rf " + output_bvm_dir_).c_str());
    unused = system(("mkdir -p " + output_bvm_dir_).c_str());

    //create output multi-level bird-view map binary file folder
    output_multi_bvm_bin_dir_ = keyframes_root_dir + "output_multi_bev/binary/";
    unused = system(("rm -rf " + output_multi_bvm_bin_dir_).c_str());
    unused = system(("mkdir -p " + output_multi_bvm_bin_dir_).c_str());

    //create output multi-level bird-view map image file folder
    output_multi_bvm_img_dir_ = keyframes_root_dir + "output_multi_bev/image/";
    unused = system(("rm -rf " + output_multi_bvm_img_dir_).c_str());
    unused = system(("mkdir -p " + output_multi_bvm_img_dir_).c_str());

    //create output single-level bird-view map csv file folder
    output_single_bvm_csv_dir_ = keyframes_root_dir + "output_single_bev/csv/";
    unused = system(("rm -rf " + output_single_bvm_csv_dir_).c_str());
    unused = system(("mkdir -p " + output_single_bvm_csv_dir_).c_str());

    //create output single-level bird-view map image file folder
    output_single_bvm_img_dir_ = keyframes_root_dir + "output_single_bev/image/";
    unused = system(("rm -rf " + output_single_bvm_img_dir_).c_str());
    unused = system(("mkdir -p " + output_single_bvm_img_dir_).c_str());
}

void setNeighbors()
{
    std::pair<int, int> neighbor;
    neighbor.first = -1; neighbor.second =  0;
    four_neighbor_iterator_.push_back(neighbor);
    neighbor.first =  0; neighbor.second =  1;
    four_neighbor_iterator_.push_back(neighbor);
    neighbor.first =  0; neighbor.second = -1;
    four_neighbor_iterator_.push_back(neighbor);
    neighbor.first =  1; neighbor.second =  0;
    four_neighbor_iterator_.push_back(neighbor);
}


/**
 * Organize the points within a cloud to their cylindrical projection order. 
 * The row_idx, col_idx must be specified in the fields of the cloud before input.
 * @param input_cloud {pcl::PointCloud<pcl::PointXYZIRCT>::Ptr}
 * @param output_cloud {pcl::PointCloud<pcl::PointXYZIRCT>::Ptr}
 * @return {*}
 */
void getOrderedCloud(
        pcl::PointCloud<pcl::PointXYZIRCT>::Ptr &input_cloud,
        pcl::PointCloud<pcl::PointXYZIRCT>::Ptr &output_cloud)
{
    output_cloud->resize(sensor_params_.N_SCAN * sensor_params_.Horizon_SCAN);

    // int numberOfCores = 8;
    // #pragma omp parallel for num_threads(numberOfCores)
    for (auto &point : input_cloud->points) {
        int row_idx = point.row;
        int col_idx = point.col;

        if (row_idx < 0 || row_idx >= sensor_params_.N_SCAN) {
            continue;
        }
        if (col_idx < 0 || col_idx >= sensor_params_.Horizon_SCAN) {
            continue;
        }

        int point_idx = row_idx * sensor_params_.Horizon_SCAN + col_idx;

        output_cloud->points[point_idx] = point;
    }
}

void markGroundPoints(
        pcl::PointCloud<pcl::PointXYZIRCT>::Ptr &output_cloud,
        cv::Mat &ground_mat)
{
    ground_mat = cv::Mat::zeros(sensor_params_.N_SCAN, sensor_params_.Horizon_SCAN, CV_8S);

    size_t lowerInd, upperInd;
    float diffX, diffY, diffZ, angle;

    //std::sort(output_cloud->points.begin(), output_cloud->points.end(), [](pcl::PointXYZIRCT &p1, pcl::PointXYZIRCT &p2) -> bool {
    //    return (p1.row < p2.row) || (p1.row == p2.row && p1.col < p2.col);
    //});

    // used to compute average ground height
    cv::Mat ground_grid_avg_heights = cv::Mat::zeros(
            GROUND_HEIGHT_GRID_ROWS, GROUND_HEIGHT_GRID_COLS, CV_32F);
    cv::Mat num_ground_grid_points = 0.01 * cv::Mat::ones(
            GROUND_HEIGHT_GRID_ROWS, GROUND_HEIGHT_GRID_COLS, CV_32F);

    // only do ground extraction within beams GROUND_UPPER_SCAN ~ N_SCAN-1
    for (int col_idx = 0; col_idx < sensor_params_.Horizon_SCAN; col_idx ++) {
        for (int row_idx = sensor_params_.N_SCAN - 1; row_idx > sensor_params_.N_SCAN - sensor_params_.GROUND_UPPER_SCAN - 1; row_idx --) {

            lowerInd = row_idx * sensor_params_.Horizon_SCAN + col_idx;
            upperInd = (row_idx - 1) * sensor_params_.Horizon_SCAN + col_idx;

            // 防止正上方有一个地面点没有读数，使用相邻点替代
            if (output_cloud->points[upperInd].intensity == -1) {
                int tmp_col_idx = (col_idx + 2) % sensor_params_.Horizon_SCAN;
                upperInd = (row_idx - 1) * sensor_params_.Horizon_SCAN + tmp_col_idx;
            }

            if (output_cloud->points[upperInd].intensity == -1) {
                int tmp_col_idx = (col_idx - 2) % sensor_params_.Horizon_SCAN;
                upperInd = (row_idx - 1) * sensor_params_.Horizon_SCAN + tmp_col_idx;
            }

            //use point on the other ring
            if (output_cloud->points[upperInd].intensity == -1 && row_idx >= 2) {
                int tmp_row_idx = row_idx - 2;
                upperInd = tmp_row_idx * sensor_params_.Horizon_SCAN + col_idx;
            }

            if (output_cloud->points[lowerInd].intensity == -1 ||
                output_cloud->points[upperInd].intensity == -1){
                // no info to check, invalid points
                ground_mat.at<int8_t>(row_idx, col_idx) = -1;
                continue;
            }

            diffX = output_cloud->points[upperInd].x - output_cloud->points[lowerInd].x;
            diffY = output_cloud->points[upperInd].y - output_cloud->points[lowerInd].y;
            diffZ = output_cloud->points[upperInd].z - output_cloud->points[lowerInd].z;

            angle = atan2(diffZ, sqrt(diffX*diffX + diffY*diffY) ) * 180.0 / M_PI;

            float sensorMountAngle = 0.0f;
            // float angle_thres = 0.4 * range_mat_.at<float>(row_idx, col_idx) + 8.8;

            // mark as ground points
            if (abs(angle - sensorMountAngle) <= 10.0f){
                ground_mat.at<int8_t>(row_idx, col_idx) = 1;
                ground_mat.at<int8_t>(row_idx - 1, col_idx) = 1;
            }
        }
    }

    // 分块求地面高度平均值
    for (int row_idx = 0; row_idx < sensor_params_.N_SCAN; row_idx ++) {
        for (int col_idx = 0; col_idx < sensor_params_.Horizon_SCAN; col_idx ++) {
            if (ground_mat.at<int8_t>(row_idx, col_idx) != 1) {
                continue;
            }
            int sector_row = 0;
            int sector_col = 0;
            int point_index = row_idx * sensor_params_.Horizon_SCAN + col_idx;
            std::tie(sector_row, sector_col) = getBelongingGrid(
                    output_cloud,
                    point_index);
            ground_grid_avg_heights.at<float>(sector_row, sector_col) +=
                    output_cloud->points[point_index].z;
            //min height instead
            //if (output_cloud->points[point_index].z < ground_grid_avg_heights.at<float>(sector_row, sector_col)) {
            //    ground_grid_avg_heights.at<float>(sector_row, sector_col) = output_cloud->points[point_index].z;
            //}

            num_ground_grid_points.at<float>(sector_row, sector_col) =
                    num_ground_grid_points.at<float>(sector_row, sector_col) + 1;
        }
    }

    ground_grid_avg_heights = ground_grid_avg_heights / num_ground_grid_points;
    // std::cout << "ground_sector_height: \n" << ground_grid_avg_heights_ << std::endl;

    // extract ground cloud (ground_mat_ == 1)
    // mark entry that doesn't need to label (ground and invalid point) for segmentation
    // note that ground remove is from 0~N_SCAN-1, need range_mat_ for mark label matrix for the 16th scan
    for (int row_idx = 0; row_idx < sensor_params_.N_SCAN; row_idx ++) {
        for (int col_idx = 0; col_idx < sensor_params_.Horizon_SCAN; col_idx ++) {

            // 防止车顶被当作地面
            int sector_row = 0;
            int sector_col = 0;
            int point_index = row_idx * sensor_params_.Horizon_SCAN + col_idx;
            std::tie(sector_row, sector_col) = getBelongingGrid(output_cloud, point_index);

            int neighbor_sector_row = 0;
            int neighbor_sector_col = 0;
            for (auto iter: four_neighbor_iterator_) {
                neighbor_sector_row = sector_row + iter.first;
                neighbor_sector_col = sector_col + iter.second;

                if (neighbor_sector_row < 0 || neighbor_sector_row >= 75 ||
                    neighbor_sector_col < 0 || neighbor_sector_col >= 50) {
                    continue;
                }
                // 检查高度：是否比周围地面分块的高度高很多
                if (output_cloud->points[point_index].z -
                    ground_grid_avg_heights.at<float>(neighbor_sector_row, neighbor_sector_col) > 0.30) {
                    ground_mat.at<int8_t>(row_idx, col_idx) = 0;
                    break;
                }
            }

            // ground points are labeled 0 in label_mat
            if (ground_mat.at<int8_t>(row_idx, col_idx) == 1){
                output_cloud->points[point_index].label = 0; // 0 means gound points
            } else {
                //output_cloud->points[point_index].label = 1; // 1 means non-gound points
            }
        }
    }

}

/**
 * @description: compute and save the multi-layer occupancy BEV images for a single point cloud. 
 * @param cloud {pcl::PointCloud<pcl::PointXYZIRCT>::Ptr}  
 * @param str_cloud_idx {std::string}  
 * @param interval=1.0f {float} 
 * @return {*}
 */
void computeAndSaveMultiBev(
    pcl::PointCloud<pcl::PointXYZIRCT>::Ptr cloud, 
    std::string str_cloud_idx, 
    float interval = 1.0f)
{
    static int MAX_RANGE = 112;
    static int MAT_SIZE = MAX_RANGE*2 / interval;
    static int NUM_BEV_LAYERS = 24; // 一共24层
    static float LIDAR_TO_GROUND_HEIGHT = 2.0f;

    std::vector<cv::Mat> multi_bev(24);
    cv::Mat tmp = cv::Mat::zeros(MAT_SIZE, MAT_SIZE, CV_8UC1);
    for (size_t layer_idx = 0; layer_idx < multi_bev.size(); layer_idx ++) {
        multi_bev[layer_idx] = tmp.clone();
    }
    
    
    for (auto &pi : cloud->points) {
        int x = round((pi.x + MAX_RANGE) / interval + 0.5);
        int y = round((pi.y + MAX_RANGE) / interval + 0.5);
        int layer_idx = round(pi.z / sensor_params_.HEIGHT_RES + LIDAR_TO_GROUND_HEIGHT);

        // skip points out side of the range
        if (x < 0 || x >= MAT_SIZE || y < 0 || y >= MAT_SIZE || layer_idx < 0 || layer_idx >= NUM_BEV_LAYERS
            || pi.label == 0) {
            continue;
        }

        if (multi_bev[layer_idx].at<uint8_t>(x, y) == 0) {
            multi_bev[layer_idx].at<uint8_t>(x, y) = 255;
        }
    }

    // save binary file (multi-layer bev in one file)
    std::string bev_bin_filename = fmt::format("{}{}.bin", output_multi_bvm_bin_dir_, str_cloud_idx);
    std::ofstream f_bev_bin(bev_bin_filename, std::ofstream::binary);
    if (!f_bev_bin.is_open()) {
        std::cerr << "Can not open file: " << bev_bin_filename << "\n";
    }

    // save image files (onr single-channel image for each layer)
    std::string img_dir = fmt::format("{}{}/", output_multi_bvm_img_dir_, str_cloud_idx);
    if (access(img_dir.c_str(), 0) == -1) {
        // mkdir if the img folder for this point-cloud does not exists
        int _ __attribute__((unused)) = system(("mkdir -p " + img_dir).c_str());
    } 
    for (int layer_idx = 0; layer_idx < (int)multi_bev.size(); layer_idx ++) {
        auto& bev_img = multi_bev[layer_idx];
        for (int row_idx = 0; row_idx < bev_img.rows; row_idx ++) {
            f_bev_bin.write(
                    reinterpret_cast<const char*>(
                            bev_img.ptr(row_idx)), 
                            bev_img.cols * bev_img.elemSize());
        }

        std::string png_filename = 
                fmt::format("{}{:02d}.png", img_dir, layer_idx);
        cv::imwrite(png_filename, bev_img);
    }
    f_bev_bin.close();
}


/**
 * @description: compute and save the single-layer height BEV images for a single point cloud. 
 * @param cloud {pcl::PointCloud<pcl::PointXYZIRCT>::Ptr}  
 * @param str_cloud_idx {std::string}  
 * @param interval=1.0f {float} 
 * @return {*}
 */
void computeAndSaveSingleBev(
    pcl::PointCloud<pcl::PointXYZIRCT>::Ptr cloud, 
    std::string str_cloud_idx, 
    float interval = 1.0f)
{
    static int MAX_RANGE = 112;
    static int MAT_SIZE = MAX_RANGE*2 / interval;
    static float LIDAR_TO_GROUND_HEIGHT = 2.0f;

    cv::Mat single_bev = cv::Mat::zeros(MAT_SIZE, MAT_SIZE, CV_8UC1);
    
    for (auto &pi : cloud->points) {
        int x = round((pi.x + MAX_RANGE) / interval + 0.5);
        int y = round((pi.y + MAX_RANGE) / interval + 0.5);
        int height = int((pi.z + LIDAR_TO_GROUND_HEIGHT)*4.0); // usually max height does not excced 50m, scale to better visualize
        height = std::min(std::max(0, height), 255);

        // skip points out side of the range
        if (x < 0 || x >= MAT_SIZE || y < 0 || y >= MAT_SIZE || pi.label == 0) {
            continue;
        }

        if (single_bev.at<uint8_t>(x, y) < height) {
            single_bev.at<uint8_t>(x, y) = height;
        }
    }

    // export png image
    std::string png_filename = 
            fmt::format("{}{}.png", output_single_bvm_img_dir_, str_cloud_idx);
    cv::imwrite(png_filename, single_bev);


    // export csv format
    std::string csv_filename = 
            fmt::format("{}{}.csv", output_single_bvm_csv_dir_, str_cloud_idx);
    std::ofstream f_csv(csv_filename);
    if (!f_csv) {
        std::cerr << "Faied to export csv formatted BEV file: " << csv_filename;
    }
    f_csv << cv::format(single_bev, cv::Formatter::FMT_CSV);
    f_csv.close();
}


/**
 * Read 6-DoF pose of each key frame from the csv file, and return them. 
 * @param pose_filename {string} 
 * @return {vector<Pose6f>} vector of poses
 */
std::vector<Pose6f> readKeyframePose(std::string pose_filename)
{
    std::fstream f_gt_pose;
    f_gt_pose.open(pose_filename, ios::in);

    if (f_gt_pose.is_open()) {
        std::cout << "loaded keyframe pose file: " << pose_filename << std::endl;
    } else {
        std::cerr << "failed to load keyframe pose file: " << pose_filename << std::endl;
        exit(1);
    }

    std::vector<Pose6f> keyframe_pose;

    // NOTE: each entry should have format of:
    /*
    cloud_idx, x, y, z, roll, pitch, yaw, \
    rotation_matrix(0 0), rotation_matrix(0 1), rotation_matrix(0 2), \
    rotation_matrix(1 0), rotation_matrix(1 1), rotation_matrix(1 2), \
    rotation_matrix(2 0), rotation_matrix(2 1), rotation_matrix(2 2)
    */ 

    std::string entry_str;
    while (f_gt_pose >> entry_str) {
        Eigen::Matrix<double,4,4> T = Eigen::Matrix<double,4,4>::Zero();
        T(3,3) = 1.0;

        std::vector<std::string> entry_tokens;

        std::stringstream  ss(entry_str);
        std::string str;
        while (getline(ss, str, ',')) {
            entry_tokens.push_back(str);
        }
        if(entry_tokens.size() != 16) {
            std::string err_msg = fmt::format("Size of entry_token is: {}, while expecting 16. ", entry_tokens.size());
            std::cerr << err_msg << std::endl;
            break;
        }
        int64_t cloud_idx;
        std::istringstream (entry_tokens[0]) >> cloud_idx;

        // translation
        T(0,3) = std::stod(entry_tokens[1]);
        T(1,3) = std::stod(entry_tokens[2]);
        T(2,3) = std::stod(entry_tokens[3]);

        // rotation
        for(int i=0; i<3; i++){
            for(int j=0; j<3; j++){
                T(i,j) = std::stod(entry_tokens[7+(3*i)+j]);
            }
        }

        //Eigen::Affine3d this_pose(T);
        Eigen::Matrix3d rotation_matrix = T.block<3,3>(0,0);
        Eigen::Quaterniond rotation_quat(rotation_matrix);
        // do NOT use the built-in conversion to euler angles in Eigen
        // Eigen::Vector3d euler_angles = rotation_matrix.eulerAngles(2,1,0); // ZYX order: yaw pitch roll
        Eigen::Vector3d euler_angles = rotationMatrixToEulerAngles(rotation_matrix);
        Pose6f this_pose6f{
                .x = float(T(0,3)),
                .y = float(T(1,3)),
                .z = float(T(2,3)),
                .roll = float(euler_angles(0)),
                .pitch = float(euler_angles(1)),
                .yaw = float(euler_angles(2)),
                .rotation_matrix = rotation_matrix,
                .rotation_quat = rotation_quat
        };

        keyframe_pose.emplace_back(std::move(this_pose6f));
    }

    f_gt_pose.close();

    std::cout << "Finish reading all keyframe pose, total " << keyframe_pose.size() << " entries. " << std::endl;

    return keyframe_pose;
}


/**
 * List all pcd files within the specified folder. 
 * @param path {string}
 * @param filenames {vector<string>}
 * @return {*}
 */
void getPcdFileNames(std::string path, std::vector<std::string>& filenames)
{
    DIR *pDir;
    struct dirent* ptr;
    if(!(pDir = opendir(path.c_str()))){
        std::cerr << "Folder doesn't Exist!" << std::endl;
        return;
    }
    while((ptr = readdir(pDir))!=0) {
        std::string short_name = ptr->d_name;
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") == 0) {
            continue;
        }
        if (short_name.substr(short_name.find_last_of('.') + 1) != "pcd") {
            continue;
        }
        if (path.back() == '/') {
            filenames.push_back(path + ptr->d_name);
        } else {
            filenames.push_back(path + "/" + ptr->d_name);
        }
    }
    closedir(pDir);

    std::sort(filenames.begin(), filenames.end());
}


/**
 * Given a set of key frames, select major frames based on their consecutive distance. 
 * @param keyframe_pose {vector<Pose6f>&}
 * @return {vector<int32_t>} the index of each major frame in the key frames order
 */
std::vector<int32_t> selectMajorFrames(std::vector<Pose6f>& keyframe_pose)
{
    static const float MAJOR_FRAME_INTERVAL = 20.0f;
    static const size_t NUM_CANDIDATES_FROM_TREE = 1; // only check on the nearest neighbor

    std::vector<int32_t> majorframe_indices;
    std::vector<std::vector<float>> major_pose_to_search;

/*
    // convert all poses to 3d vec that kd tree could handle
    std::for_each(keyframe_pose.begin(), keyframe_pose.end(), 
            [&major_pose_all](Pose6f& this_pose) -> void {
                major_pose_all.emplace_back(std::move(this_pose.getPositionVec()));
            });
*/
    // 0th frame is the first major frame
    Pose6f& last_major_pose = keyframe_pose[0];
    majorframe_indices.push_back(0);
    major_pose_to_search.push_back(last_major_pose.getPositionVec());

    for (int frame_idx = 1; frame_idx < (int)keyframe_pose.size(); frame_idx ++) {
        Pose6f& last_major_pose = keyframe_pose[majorframe_indices.back()];
        Pose6f& this_frame_pose = keyframe_pose[frame_idx];
        
        // Step 1: check distance to last consecutive major frame
        float dist_to_last_major = getDistance(this_frame_pose, last_major_pose);
        if (dist_to_last_major < MAJOR_FRAME_INTERVAL) {
            // early skip
            continue;
        }

        // Step 2: check distance to all previous major frames
        std::unique_ptr<InvKeyTree> pose_tree = std::make_unique<InvKeyTree>(
            3 /* dim */,
            major_pose_to_search,
            10 /* max leaf */
        );

        // knn search
        std::vector<size_t> candidate_indexes(NUM_CANDIDATES_FROM_TREE);
        std::vector<float> out_dists_sqr(NUM_CANDIDATES_FROM_TREE);

        nanoflann::KNNResultSet<float> knnsearch_result(NUM_CANDIDATES_FROM_TREE);
        knnsearch_result.init(&candidate_indexes[0], &out_dists_sqr[0]);
        auto this_key_position = this_frame_pose.getPositionVec();
        pose_tree->index->findNeighbors(
            knnsearch_result,
            this_key_position.data(), // ptr to query vec
            nanoflann::SearchParams(10));
        // 搜索到的candidate index存储在candidate_indexes中
        if (out_dists_sqr[0] < MAJOR_FRAME_INTERVAL * MAJOR_FRAME_INTERVAL) {
            std::string msg = fmt::format("Key Frame {} overlaps with previous Major Frame {}, i.e. Key Frame {}. \n", 
                    frame_idx, candidate_indexes[0], majorframe_indices[candidate_indexes[0]]);
            std::cout << msg;
            
            continue;
        }

        // Step 3: save this major frame
        majorframe_indices.push_back(frame_idx);
        major_pose_to_search.push_back(this_key_position);
    }

    return majorframe_indices;
}


/**
 * Get smoothed labels for all keyframes, interpolated on the integer labels created for major frames. 
 * @param key_frame_poses {std::vector<Pose6f>&}
 * @param major_frame_indeices {std::vector<int32_t>&}
 * @return {vector<LabelType>} vector of smoothed labels for all key frames
 */
std::vector<LabelType> getKeyFrameLabel(std::vector<Pose6f>& key_frame_poses, std::vector<int32_t>& major_frame_indeices)
{
    static const size_t NUM_CANDIDATES_FROM_TREE = 2; // find 2 nearest major-frame neighbors
    std::vector<LabelType> key_frame_labels(key_frame_poses.size(), LabelType(major_frame_indeices.size(), 0));
    
    std::string msg = fmt::format("One-hot label has length: {:d}", major_frame_indeices.size());
    std::cout << msg << std::endl;
    
    // Step 1: build kd tree for nearest major frame search
    // convert poses of all major frames to 3d vec that kd tree could handle
    std::vector<std::vector<float>> major_pose_to_search;
    std::for_each(
            major_frame_indeices.begin(), major_frame_indeices.end(), 
            [&major_pose_to_search, &key_frame_poses](int32_t major_frame_index) -> void {
                major_pose_to_search.emplace_back(
                        std::move(key_frame_poses[major_frame_index].getPositionVec()));
            });
    // create tree of all major frames
    std::unique_ptr<InvKeyTree> pose_tree = std::make_unique<InvKeyTree>(
        3 /* dim */,
        major_pose_to_search,
        10 /* max leaf */
    );


    // Step 2: traverse all keyframes and compute the smoothed label using its nearest major-frame neighbors
    for (int key_frame_idx = 0; key_frame_idx < (int)key_frame_poses.size(); key_frame_idx ++) {
        auto& this_frame_pose = key_frame_poses[key_frame_idx];

        std::vector<size_t> candidate_indexes(NUM_CANDIDATES_FROM_TREE);
        std::vector<float> out_dists_sqr(NUM_CANDIDATES_FROM_TREE);

        nanoflann::KNNResultSet<float> knnsearch_result(NUM_CANDIDATES_FROM_TREE);
        knnsearch_result.init(&candidate_indexes[0], &out_dists_sqr[0]);
        auto this_key_position = this_frame_pose.getPositionVec();
        pose_tree->index->findNeighbors(
            knnsearch_result,
            this_key_position.data(), // ptr to query vec
            nanoflann::SearchParams(10));
        
        // check nearest candidates
        if (key_frame_idx == major_frame_indeices[candidate_indexes[0]]) {
            // this key frame itself is a major frame
            key_frame_labels[key_frame_idx][candidate_indexes[0]] = 1.0f;
        
        } else {
            // interpolate on the nearest two major labels
            // NOTE: this operation is heuristic
            float weight_0 = 1.0f / (out_dists_sqr[0] + 1e-5);
            float weight_1 = 1.0f / (out_dists_sqr[1] + 1e-5);
            float sum_weights = weight_0 + weight_1;
            weight_0 /= sum_weights;
            weight_1 /= sum_weights;

            key_frame_labels[key_frame_idx][candidate_indexes[0]] = weight_0;
            key_frame_labels[key_frame_idx][candidate_indexes[1]] = weight_1;
        }
    }

    return key_frame_labels;

}


/**
 * Save labels for all key frames to a csv file
 * @param key_frame_labels {vector<LabelType>} 
 * @param label_filename {string} 
 * @return {*}
 */
void saveLabels(std::vector<LabelType> key_frame_labels, std::string label_filename)
{
    std::ofstream f_labels(label_filename);
    if (!f_labels.is_open()) {
        std::cerr << "failed to open keyframe label file: " << label_filename << std::endl;
        exit(1);
    }

    for (LabelType& label : key_frame_labels) {
        std::ostream_iterator<float> itr(f_labels, ",");
        std::copy(label.begin(), label.end(), itr);
        f_labels << "\n";
    }

    std::string msg = fmt::format("saved labels from {:d} key frames. ", key_frame_labels.size());
    std::cout << msg << std::endl;
}


int main(int argc, char** argv)
{
    if (argv[1] == nullptr || argv[2] == nullptr) {
        std::string usage_prompt = fmt::format(
"\
Usage: {} [keyframes_root_dir] [sensor_type]\n\n\
[keyframes_root_dir] should be organized as follows: \n\
[keyframes_root_dir]\n\
├ keyframe_point_cloud/ <- folder for selected point clouds in pcd format for each frame \n\
├ keyframe_pose.csv <- 6-DoF pose for each frame \n\
└ keyframe_pose_format.csv <- 6-DoF pose format description \n\
\n\
[sensor_type] could be HDL_32E, HDL_64E or OS1_64. \n\
\n\
This binary generates ground-removed point clouds, \
single & multi layer BEV images and creates geometric distance-based labels for each point cloud. \
After running the binary, you will have files organized as follows: \n \
[keyframes_root_dir]\n \
├ ... \n \
├ non_ground_point_cloud/ <- folder for ground-removes point clouds in pcd format \n \
├ output_multi_bev/ <- folder for multi-layer BEV images \n \
└ output_single_bev <- folder for single-layer BEV images \n \
", argv[0]);
        std::cout << usage_prompt << std::endl;
        exit(1);
    }
    std::string keyframes_root_dir(argv[1]);
    if (keyframes_root_dir.back() != '/') {
        keyframes_root_dir.append("/");
    }
    // input dir
    std::string keyframes_point_cloud_dir = keyframes_root_dir + "keyframe_point_cloud/";
    // output dir
    std::string non_ground_point_cloud_dir = keyframes_root_dir + "non_ground_point_cloud/";
    // input file
    std::string keyframes_pose_file = keyframes_root_dir + "keyframe_pose.csv";
    // output file
    std::string keyframes_label_file = keyframes_root_dir + "keyframe_label.csv";

    int unused __attribute__((unused));
    unused = system(("rm -rf " + non_ground_point_cloud_dir).c_str());
    unused = system(("mkdir -p " + non_ground_point_cloud_dir).c_str());

    // get pcd filenames under the specified dir
    std::vector<std::string> all_pcd_files;
    getPcdFileNames(keyframes_point_cloud_dir, all_pcd_files);

    // set neighbors offset indices for bfs
    setNeighbors();

    // init globally used directories
    initDirectories(keyframes_root_dir);
    
    // parse and set sensor params
    SensorType sensor_type = parseSensorType(std::string(argv[2]));
    sensor_params_ = getSensorParams(sensor_type);
    std::string sensor_msg = fmt::format("Using sensor_type {}, with params: {}\n", 
            std::string(argv[2]), printSensorParams(sensor_params_));
    std::cout << sensor_msg;

    double total_tiempo_ms = 0;

    // Step 1: convert all point clouds to bird-view map
    for (auto &input_filename : all_pcd_files) {
        pcl::PointCloud<pcl::PointXYZIRCT>::Ptr cloud_unordered(new pcl::PointCloud<pcl::PointXYZIRCT>());
        pcl::PointCloud<pcl::PointXYZIRCT>::Ptr cloud_ordered(new pcl::PointCloud<pcl::PointXYZIRCT>());
        pcl::io::loadPCDFile(input_filename, *cloud_unordered);

        auto time_start = std::chrono::system_clock::now();

        cv::Mat ground_mat;
        getOrderedCloud(cloud_unordered, cloud_ordered);
        markGroundPoints(cloud_ordered, ground_mat);

        float interval_res = 1.0f;
        int start_pos = input_filename.find_last_of('/') + 1;
        int end_pos = input_filename.find_last_of('.') - 1;
        int name_length = end_pos - start_pos + 1;
        std::string short_name = input_filename.substr(start_pos, name_length);

        std::cout << "Converting file: " << short_name << "\n";
        //save bird view map, csv format
        computeAndSaveMultiBev(cloud_ordered, short_name, interval_res);
        computeAndSaveSingleBev(cloud_ordered, short_name, interval_res);

        auto time_end = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
        // std::cout << "[TIME] Preprocessing and BEV generation: " << duration.count() * 1e-3 << "ms. \n";
        total_tiempo_ms += duration.count() * 1e-3;

        //save ground-removed point cloud
        std::string pcd_filename = non_ground_point_cloud_dir + short_name + ".pcd";
        pcl::io::savePCDFileBinary(pcd_filename, *cloud_ordered);
    }

    std::cout << "[TIME] Average preprocessing and BEV generation: " << total_tiempo_ms / all_pcd_files.size() << "\n";
    
    // Step 2: select major frames
    std::vector<Pose6f> keyframe_poses = readKeyframePose(keyframes_pose_file);
    std::vector<int32_t> major_frame_indices = selectMajorFrames(keyframe_poses);
    std::vector<LabelType> key_frame_labels = getKeyFrameLabel(keyframe_poses, major_frame_indices);
    saveLabels(key_frame_labels, keyframes_label_file);


    std::cout << "Done. " << std::endl;

    return 0;
}