cmake_minimum_required(VERSION 3.8)
project(openflex_isaac_contract)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS scripts/verify_embodiment_contract.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
