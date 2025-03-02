cmake_minimum_required(VERSION 3.10)
project(EnterpriseAIBridge CXX)

# C++ standard settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Output information about the environment
message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_SYSTEM_VERSION: ${CMAKE_SYSTEM_VERSION}")
message(STATUS "CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMAKE_CXX_COMPILER_VERSION: ${CMAKE_CXX_COMPILER_VERSION}")

# Find CURL package - Not required for compilation
set(CURL_NO_CURL_CMAKE ON)  # Skip CURL's own CMake config

# Define paths for Windows
if(WIN32)
    # Try different common paths for CURL on Windows
    set(CURL_INCLUDE_SEARCH_PATHS
        "C:/Program Files/curl/include"
        "C:/curl/include"
        "C:/Libraries/curl/include"
        "C:/vcpkg/installed/x64-windows/include"
        "C:/vcpkg/installed/x86-windows/include"
        "C:/msys64/mingw64/include"
    )
    
    set(CURL_LIBRARY_SEARCH_PATHS
        "C:/Program Files/curl/lib"
        "C:/curl/lib"
        "C:/Libraries/curl/lib"
        "C:/vcpkg/installed/x64-windows/lib"
        "C:/vcpkg/installed/x86-windows/lib"
        "C:/msys64/mingw64/lib"
    )
    
    # Try to find curl.h
    foreach(path ${CURL_INCLUDE_SEARCH_PATHS})
        if(EXISTS "${path}/curl/curl.h")
            set(CURL_INCLUDE_DIR "${path}")
            message(STATUS "Found CURL headers at: ${CURL_INCLUDE_DIR}")
            break()
        endif()
    endforeach()
    
    # Try to find libcurl.lib or curl.lib
    foreach(path ${CURL_LIBRARY_SEARCH_PATHS})
        if(EXISTS "${path}/libcurl.lib")
            set(CURL_LIBRARY "${path}/libcurl.lib")
            message(STATUS "Found CURL library at: ${CURL_LIBRARY}")
            break()
        elseif(EXISTS "${path}/curl.lib")
            set(CURL_LIBRARY "${path}/curl.lib")
            message(STATUS "Found CURL library at: ${CURL_LIBRARY}")
            break()
        endif()
    endforeach()
    
    # Set CURL_FOUND if we found both headers and libraries
    if(CURL_INCLUDE_DIR AND CURL_LIBRARY)
        set(CURL_FOUND TRUE)
    else()
        set(CURL_FOUND FALSE)
    endif()
else()
    # On Unix-like systems, try the standard FindCURL module
    find_package(CURL QUIET)
endif()

# If CURL was found, use it, otherwise compile in mock/stub mode
if(CURL_FOUND)
    include_directories(${CURL_INCLUDE_DIR})
    message(STATUS "Building with CURL support")
    add_definitions(-DCURL_ENABLED)
else()
    message(WARNING "CURL not found. Building with HTTP stub implementation.")
    add_definitions(-DCURL_DISABLED)
endif()

# Find nlohmann_json for JSON handling
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    message(STATUS "nlohmann_json not found, downloading via FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.10.5
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# Add source files
add_executable(api_bridge
    src/utils/api_bridge.cpp
)

# Include directories
target_include_directories(api_bridge PRIVATE
    src
    src/utils
)

# Link libraries
if(CURL_FOUND)
    target_link_libraries(api_bridge PRIVATE
        ${CURL_LIBRARY}
        nlohmann_json::nlohmann_json
    )
else()
    target_link_libraries(api_bridge PRIVATE
        nlohmann_json::nlohmann_json
    )
endif()

# Optional - Link with LibTorch if available
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../lib/libtorch")
find_package(Torch QUIET)
if(Torch_FOUND)
    message(STATUS "Torch found at ${TORCH_INSTALL_PREFIX}")
    target_include_directories(api_bridge PRIVATE ${TORCH_INCLUDE_DIRS})
    target_link_libraries(api_bridge PRIVATE ${TORCH_LIBRARIES})
    target_compile_definitions(api_bridge PRIVATE TORCH_AVAILABLE)
else()
    message(STATUS "Torch not found, building API bridge without LibTorch integration")
endif()

# Installation
install(TARGETS api_bridge DESTINATION bin)

# Print summary
message(STATUS "Configuration summary:")
message(STATUS "  CURL support: ${CURL_FOUND}")
message(STATUS "  JSON support: ${nlohmann_json_FOUND}")
message(STATUS "  Torch support: ${Torch_FOUND}")