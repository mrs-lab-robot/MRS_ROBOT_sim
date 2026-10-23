# VR 遥操作单体测试

一次只测一个执行器（空场景 `openflex_empty_scene`），不需要人或头显参与。

前提：仿真 + ROS 桥都在跑（5100 有 `pico_pose_bridge`、24102 有仿真 teleop）。

```bash
# 需要 ROS 环境
source /opt/ros/humble/setup.bash
source <openflex_ws>/install/setup.bash
export ROS_DOMAIN_ID=0

python3 sim_runtime/ros2/tests/vr_teleop_case.py --case arms
python3 sim_runtime/ros2/tests/vr_teleop_case.py --case head
python3 sim_runtime/ros2/tests/vr_teleop_case.py --case base
python3 sim_runtime/ros2/tests/vr_teleop_case.py --case lift
python3 sim_runtime/ros2/tests/vr_teleop_case.py --case hold   # 静默 10s 看重力漂移
```

每个 case 只驱动对应的一路，输出 `CASE <name>: PASS/FAIL | <数字>`：
- arms：双臂 12s 最大摆幅 > 0.05 rad
- head：按一次 B + 头姿摆动，头部摆幅 > 0.05 rad
- base：左摇杆推到底，`/cmd_vel` 峰值 > 0.05 m/s
- lift：按住 Y 8s，升降 delta > +0.02 rad
- hold：静默 10s，升降漂移 |drift| < 0.005 rad
