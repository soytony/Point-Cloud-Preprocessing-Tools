//
// Created by tony on 22-4-17.
//

#include "BatchCloudManip.h"


//typedef PointXYZIRCL   PointType;


const int GROUND_HEIGHT_GRID_ROWS = 75;
const int GROUND_HEIGHT_GRID_COLS = 50;
const int Horizon_SCAN = 2083; // for hdl 64e
const int N_SCAN = 64;


std::vector<std::pair<int, int>> four_neighbor_iterator_;


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

std::vector<std::string> splitString(std::string input_str, char token)
{
    std::stringstream ss(input_str);
    std::vector<std::string> result;

    std::string tmp_strlet;

    while (getline(ss, tmp_strlet, token)) {
        result.push_back(tmp_strlet);
    }

    return result;
}

void getOrderedCloud(
        pcl::PointCloud<pcl::PointXYZIRCT>::Ptr &input_cloud,
        pcl::PointCloud<pcl::PointXYZIRCT>::Ptr &output_cloud)
{
    output_cloud->resize(N_SCAN * Horizon_SCAN);

    // int numberOfCores = 8;
    // #pragma omp parallel for num_threads(numberOfCores)
    for (auto &point : input_cloud->points) {
        int row_idx = point.row;
        int col_idx = point.col;

        int point_idx = row_idx * Horizon_SCAN + col_idx;

        output_cloud->points[point_idx] = point;
    }
}

void markGroundPoints(
        pcl::PointCloud<pcl::PointXYZIRCT>::Ptr &output_cloud,
        cv::Mat &ground_mat)
{
    ground_mat = cv::Mat::zeros(N_SCAN, Horizon_SCAN, CV_8S);

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

    // FIXME: should change groundScanInd for hdl-64e
    int groundScanInd = 50;
    for (int col_idx = 0; col_idx < Horizon_SCAN; col_idx ++) {
        for (int row_idx = N_SCAN - 1; row_idx > N_SCAN -  groundScanInd - 1; row_idx --) {

            lowerInd = row_idx * Horizon_SCAN + col_idx;
            upperInd = (row_idx - 1) * Horizon_SCAN + col_idx;

            // 防止正上方有一个地面点没有读数，使用相邻点替代
            if (output_cloud->points[upperInd].intensity == -1) {
                int tmp_col_idx = (col_idx + 2) % Horizon_SCAN;
                upperInd = (row_idx - 1) * Horizon_SCAN + tmp_col_idx;
            }

            if (output_cloud->points[upperInd].intensity == -1) {
                int tmp_col_idx = (col_idx - 2) % Horizon_SCAN;
                upperInd = (row_idx - 1) * Horizon_SCAN + tmp_col_idx;
            }

            //use point on the other ring
            if (output_cloud->points[upperInd].intensity == -1 && row_idx >= 2) {
                int tmp_row_idx = row_idx - 2;
                upperInd = tmp_row_idx * Horizon_SCAN + col_idx;
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
    for (int row_idx = 0; row_idx < N_SCAN; row_idx ++) {
        for (int col_idx = 0; col_idx < Horizon_SCAN; col_idx ++) {
            if (ground_mat.at<int8_t>(row_idx, col_idx) != 1) {
                continue;
            }
            int sector_row = 0;
            int sector_col = 0;
            int point_index = row_idx * Horizon_SCAN + col_idx;
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
    for (int row_idx = 0; row_idx < N_SCAN; row_idx ++) {
        for (int col_idx = 0; col_idx < Horizon_SCAN; col_idx ++) {

            // 防止车顶被当作地面
            int sector_row = 0;
            int sector_col = 0;
            int point_index = row_idx * Horizon_SCAN + col_idx;
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

void saveAsMat(
    pcl::PointCloud<pcl::PointXYZIRCT>::Ptr cloud, 
    std::string filename_sin_appendix, 
    float interval = 2.0f)
{
    std::string mat_filename = filename_sin_appendix + ".csv";
    std::string png_filename = filename_sin_appendix + ".png";

    static int MAX_RANGE = 100;
    static int MAT_SIZE = MAX_RANGE*2 / interval + 1;
    cv::Mat cart_bv = cv::Mat::zeros(MAT_SIZE, MAT_SIZE, CV_32FC1);
    
    
    for (auto &pi : cloud->points) {
        int x = round((pi.x + MAX_RANGE) / interval + 0.5);
        int y = round((pi.y + MAX_RANGE) / interval + 0.5);

        if (x < 0 || x >= MAT_SIZE || y < 0 || y >= MAT_SIZE || pi.label == 0) {
            continue;
        }

        if (pi.z + 2.0f > cart_bv.at<float>(x, y)) {
            cart_bv.at<float>(x, y) = pi.z + 2.0f;
        }
    }

    cv::Ptr<cv::Formatter> fmt = cv::Formatter::get(cv::Formatter::FMT_CSV);
    fmt->set64fPrecision(4);
    fmt->set32fPrecision(4);

    std::ofstream of(mat_filename, std::ios::out);
    if (of.is_open()) {
        of << fmt->format(cart_bv);
    } else {
        std::cerr << "Can not open file: " << mat_filename << "\n";
    }

    cv::imwrite(png_filename, cart_bv);
}


void getPcdFileNames(std::string path, std::vector<std::string>& filenames)
{
    DIR *pDir;
    struct dirent* ptr;
    if(!(pDir = opendir(path.c_str()))){
        std::cerr << "Folder doesn't exist!" << std::endl;
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

int main(int argc, char** argv)
{
    if (argv[1] == nullptr) {
        std::cout << "Usage: " << argv[0] << " <keyframes_root_dir>" << std::endl;
        exit(1);
    }
    std::string keyframes_root_dir(argv[1]);
    std::string keyframes_point_cloud_dir = (keyframes_root_dir.back() == '/')?
            keyframes_root_dir + "keyframe_point_cloud/" : keyframes_root_dir + "/" + "keyframe_point_cloud/";

    std::string non_ground_point_cloud_dir = (keyframes_root_dir.back() == '/')?
            keyframes_root_dir + "non_ground_point_cloud/" : keyframes_root_dir + "/" + "non_ground_point_cloud/";

    int unused __attribute__((unused));
    unused = system(("rm -rf " + non_ground_point_cloud_dir).c_str());
    unused = system(("mkdir -p " + non_ground_point_cloud_dir).c_str());

    std::vector<std::string> all_pcd_files;
    getPcdFileNames(keyframes_point_cloud_dir, all_pcd_files);

    setNeighbors();

    //create output bird-view map folder
    std::string output_bvm_dir = (keyframes_root_dir.back() == '/')?
            keyframes_root_dir + "output_bvm/" : keyframes_root_dir + "/" + "output_bvm/";
    unused = system(("rm -rf " + output_bvm_dir).c_str());
    unused = system(("mkdir -p " + output_bvm_dir).c_str());

    double total_tiempo_ms = 0;

    //convert all point clouds to bird-view map
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
        saveAsMat(cloud_ordered, output_bvm_dir + short_name, interval_res);

        auto time_end = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
        std::cout << "[TIME] Preprocessing and BEV generation: " << duration.count() * 1e-3 << "ms. \n" << endl;
        total_tiempo_ms += duration.count() * 1e-3;

        //save ground-removed point cloud
        pcl::io::savePCDFileBinary(non_ground_point_cloud_dir + short_name + ".pcd", *cloud_ordered);
    }

    std::cout << "[TIME] Average preprocessing and BEV generation: " << total_tiempo_ms / all_pcd_files.size() << "\n";
    std::cout << "Done. " << std::endl;

    return 0;
}