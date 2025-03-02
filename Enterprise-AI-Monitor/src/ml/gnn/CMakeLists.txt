add_library(gnn STATIC
    network_model.cpp
)

target_include_directories(gnn PUBLIC 
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/preprocessing
)

target_link_libraries(gnn PRIVATE 
    ${TORCH_LIBRARIES}
)