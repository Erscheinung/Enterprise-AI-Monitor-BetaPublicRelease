cmake_minimum_required(VERSION 3.15)
project(EnterpriseAIMonitor LANGUAGES CXX CUDA)

# Set CUDA toolkit first
set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin/nvcc.exe")
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4")
set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4")

# Find CUDA package
find_package(CUDAToolkit REQUIRED)

# Find Python packages
find_package(Python3 COMPONENTS Development REQUIRED)
include_directories(${Python3_INCLUDE_DIRS})

# C++ standard settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set Torch path and find package
set(CMAKE_PREFIX_PATH "C:/Users/91965/Downloads/libtorch-win-shared-with-deps-2.5.1+cu124/libtorch")
find_package(Torch REQUIRED)
if(Torch_FOUND)
    message(STATUS "Torch found at ${TORCH_INSTALL_PREFIX}")
else()
    message(FATAL_ERROR "Torch not found!")
endif()

# Add include directories globally
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
)

# Ensure Torch is linked properly
add_definitions(-DTORCH_API_INCLUDE_EXTENSION_H)

# Add subdirectories for compilation
add_subdirectory(src)

# Find cURL for API communication
find_package(CURL REQUIRED)
include_directories(${CURL_INCLUDE_DIRS})

# Find nlohmann_json for JSON handling
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.10.5
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# Add the new source files to your existing targets
target_sources(your_existing_target PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/api_bridge.cpp
)

# Link the additional libraries
target_link_libraries(your_existing_target PRIVATE
    ${CURL_LIBRARIES}
    nlohmann_json::nlohmann_json
)