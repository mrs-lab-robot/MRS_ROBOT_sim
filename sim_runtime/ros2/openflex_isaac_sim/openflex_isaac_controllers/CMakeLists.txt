cmake_minimum_required(VERSION 3.8)
project(openflex_isaac_controllers)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(controller_interface REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(realtime_tools REQUIRED)
find_package(std_msgs REQUIRED)

add_library(${PROJECT_NAME} SHARED
  src/explicit_topic_position_controller.cpp
)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(${PROJECT_NAME}
  controller_interface
  hardware_interface
  pluginlib
  rclcpp
  rclcpp_lifecycle
  realtime_tools
  std_msgs
)

pluginlib_export_plugin_description_file(controller_interface explicit_topic_position_controller.xml)

ament_python_install_package(${PROJECT_NAME})

install(TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)
install(FILES explicit_topic_position_controller.xml DESTINATION share/${PROJECT_NAME})
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

ament_export_dependencies(
  controller_interface
  hardware_interface
  pluginlib
  rclcpp
  rclcpp_lifecycle
  realtime_tools
  std_msgs
)
ament_export_targets(${PROJECT_NAME}Targets HAS_LIBRARY_TARGET)
ament_package()
