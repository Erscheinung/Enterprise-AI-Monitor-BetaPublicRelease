add_library(rl STATIC
    business_model.cpp
)

target_include_directories(rl PUBLIC 
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/preprocessing
)

target_link_libraries(rl PRIVATE 
    ${TORCH_LIBRARIES}
)