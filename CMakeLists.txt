# cmake_minimum_required(VERSION 3.21)
cmake_minimum_required(VERSION 3.16)
project(pointcloud_pca_test)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall -g -ggdb -DDEBUG")

find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
FIND_PACKAGE(OpenMP REQUIRED)

if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    include
)

add_subdirectory(ThirdParties/BackwardCpp)
add_subdirectory(ThirdParties/fmt)


# Boost Package ############################################################
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.58.0 COMPONENTS program_options filesystem REQUIRED)
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} include)

add_library(pcl_norm_2d src/PCA2D.cpp src/Normal2dEstimation.cpp)
set_target_properties(pcl_norm_2d PROPERTIES PUBLIC_HEADER "include/Normal2dEstimation.h;include/PCA2D.h")
install(TARGETS pcl_norm_2d
  PUBLIC_HEADER DESTINATION include/pcl_normal_estimation_2d)


# Executables ############################################################
add_executable(pointcloud_pca_test main.cpp)

target_link_libraries(pointcloud_pca_test
        ${PCL_LIBRARIES}
)

# cloud_manip
add_executable(cloud_manip
        CloudManip.cpp
)

target_link_libraries(cloud_manip
        ${PCL_LIBRARIES}
        ${OpenCV_LIBRARIES}
)

# batch_cloud_manip
add_executable(batch_cloud_manip
        BatchCloudManip.cpp
        )

target_link_libraries(batch_cloud_manip
        ${PCL_LIBRARIES}
        ${OpenCV_LIBRARIES}
        )

# mulran_point_cloud_select
add_executable(mulran_point_cloud_select
        MulranPointCloudSelect.cpp
        src/Utility.cpp
        )

target_link_libraries(mulran_point_cloud_select
        ${PCL_LIBRARIES}
        ${OpenCV_LIBRARIES}
        )


# kitti_point_cloud_select
add_executable(kitti_point_cloud_select
        KittiPointCloudSelect.cpp
        )

target_link_libraries(kitti_point_cloud_select
        ${PCL_LIBRARIES}
        ${OpenCV_LIBRARIES}
        )

# oxford radar robotcar point cloud select
add_executable(oxford_point_cloud_select
        OxfordPointCloudSelect.cpp
        )

target_link_libraries(oxford_point_cloud_select
        ${PCL_LIBRARIES}
        ${OpenCV_LIBRARIES}
        )

# test for icp align on top part of the cloud
add_executable(top_part_registration
        TopPartRegistration.cpp
        ${BACKWARD_ENABLE})

target_link_libraries(top_part_registration
        ${PCL_LIBRARIES}
        ${OpenCV_LIBRARIES}
        dw
        pcl_norm_2d)


# batch icp align on top part of the cloud
add_executable(batch_top_part_registration
        BatchTopPartRegistration.cpp
        ${BACKWARD_ENABLE})

target_link_libraries(batch_top_part_registration
        ${PCL_LIBRARIES}
        ${OpenCV_LIBRARIES}
        dw
        pcl_norm_2d)

# batch icp align on top part of the cloud
add_executable(batch_whole_registration
        BatchWholeRegistration.cpp
        ${BACKWARD_ENABLE})

target_link_libraries(batch_whole_registration
        ${PCL_LIBRARIES}
        ${OpenCV_LIBRARIES}
        dw
        pcl_norm_2d)

# batch multi-level bev generator
add_executable(batch_multi_bev_gen
        BatchMultiBevGen.cpp
        src/Utility.cpp
        ${BACKWARD_ENABLE})

target_link_libraries(batch_multi_bev_gen
        ${PCL_LIBRARIES}
        ${OpenCV_LIBRARIES}
        fmt
        dw
        pcl_norm_2d)