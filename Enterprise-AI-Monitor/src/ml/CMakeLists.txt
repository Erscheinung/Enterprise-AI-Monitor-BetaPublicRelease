# Add subdirectories for each ML component
add_subdirectory(lstm)
add_subdirectory(gnn)
add_subdirectory(rl)

# Create the main ml library
add_library(ml INTERFACE)
target_link_libraries(ml INTERFACE
    lstm
    gnn
    rl
)