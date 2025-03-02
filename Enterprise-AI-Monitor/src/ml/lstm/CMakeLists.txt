add_library(lstm STATIC
    infrastructure_model.cpp
)

target_include_directories(lstm PUBLIC 
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/preprocessing
)

target_link_libraries(lstm PRIVATE ${TORCH_LIBRARIES})