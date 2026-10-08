cmake_minimum_required(VERSION 3.8)
project(openflex_isaac_bringup)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(lift_slide_panel REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rviz_common REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(swerve_base_panel REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Core Widgets)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_library(${PROJECT_NAME}_rviz_panel SHARED
  include/openflex_isaac_bringup/simulation_control_sensors_panel.hpp
  src/simulation_control_sensors_panel.cpp
)
target_include_directories(${PROJECT_NAME}_rviz_panel PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(${PROJECT_NAME}_rviz_panel
  lift_slide_panel
  pluginlib
  rclcpp
  rviz_common
  sensor_msgs
  swerve_base_panel
)
target_link_libraries(${PROJECT_NAME}_rviz_panel
  Qt5::Core
  Qt5::Widgets
)
pluginlib_export_plugin_description_file(rviz_common plugins/rviz_panel_plugin.xml)
ament_export_libraries(${PROJECT_NAME}_rviz_panel)
ament_export_dependencies(lift_slide_panel pluginlib rclcpp rviz_common sensor_msgs swerve_base_panel)

install(TARGETS ${PROJECT_NAME}_rviz_panel
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)
install(FILES plugins/rviz_panel_plugin.xml DESTINATION share/${PROJECT_NAME}/plugins)

if(BUILD_TESTING)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_test(
    NAME test_rviz_sensor_displays
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/test_rviz_sensor_displays.py
  )
  set_tests_properties(test_rviz_sensor_displays PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
endif()

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install config files
# Install RViz configs as files so incremental builds notice edits to an
# existing config even when the directory timestamp itself did not change.
install(FILES
  rviz/openflex_sim.rviz
  rviz/robot_control_only.rviz
  rviz/sensor_monitor.rviz
  DESTINATION share/${PROJECT_NAME}/rviz
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}/
)

install(
  PROGRAMS
    scripts/camera_contract_publisher.py
    scripts/lidar_contract_publisher.py
    scripts/vla_contract_bridge.py
    scripts/robot_description_publisher.py
    scripts/isaacsim_compat_bridge.py
    scripts/verify_camera_images.py
    scripts/start_robot_control_sim.py
    scripts/setup_loaded_robot_when_ready.py
    scripts/spawn_robot_when_ready.py
    scripts/wait_for_sim_ready.py
    scripts/verify_mobile_base_runtime.py
    scripts/verify_upper_body_runtime.py
    scripts/verify_vla_runtime.py
    scripts/run_remote_rviz_client.sh
    scripts/stop_stale_vr_processes.py
    scripts/isaacsim_vr_arm_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
