from setuptools import find_packages, setup


package_name = "mrs_robot_arena_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", [
            "launch/arena_vr_teleop.launch.py",
            "launch/ros_control.launch.py",
        ]),
        ("share/" + package_name + "/config", [
            "config/arena_vr_teleop.yaml",
            "config/ros_control.yaml",
        ]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="MRS Robot Team",
    maintainer_email="robotics@example.invalid",
    description="ROS 2 relay between existing OpenFlex VR teleoperation and Isaac Lab-Arena",
    license="Apache-2.0",
    entry_points={"console_scripts": ["arena_relay = mrs_robot_arena_bridge.relay_node:main"]},
)
