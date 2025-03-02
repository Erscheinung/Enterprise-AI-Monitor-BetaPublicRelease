add_subdirectory(ml)
add_subdirectory(core)
add_subdirectory(utils)

add_executable(enterprise_monitor main.cpp)
target_link_libraries(enterprise_monitor PRIVATE
    ml
    core
    utils
    ${TORCH_LIBRARIES}
)