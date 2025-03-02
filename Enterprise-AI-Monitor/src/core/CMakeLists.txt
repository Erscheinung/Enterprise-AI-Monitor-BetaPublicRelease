add_library(core STATIC
    core.cpp
)

target_include_directories(core PUBLIC 
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(core PRIVATE ${TORCH_LIBRARIES})