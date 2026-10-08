cmake_minimum_required(VERSION 3.8)
project(openflex_isaac_description)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

# Install URDF files
install(DIRECTORY urdf
  DESTINATION share/${PROJECT_NAME}/
)

install(
  PROGRAMS
    scripts/generate_isaac_urdf.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install mesh files
install(DIRECTORY meshes
  DESTINATION share/${PROJECT_NAME}/
)

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
