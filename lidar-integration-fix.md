# 雷达集成链路收口——direct_poll 移植框架（工作日战）

> 写入 2026-10-14 工作日。落盘说明：任务书指定 `.hermes/lidar-integration-fix.md`，该目录为 protected agent-instruction file（同 lidar-firstframe-final.md 先例），故按既有预案落项目根目录同名文件。
> 上棒 stable-rebuild 的集成链路 ARENA_READY 16s 正常、MID360 创建成功，但 PointCloud2 不过 relay + Warp CUDA 700 ×N（Buffer.h:469）。裸探针同卡同配置实证出真点（MID360_APPROX→multiScan100/multiScan165/Profile01_20Hz_0p5deg，tick 10Hz，第 6 帧 481 点起，tsNs 正常推进）。本框架把裸探针的"出点读数"搬进集成，绕开 helper graph 与 writer 注册表。

## 一、根因判定（不重查的证据链）

| 事实 | 证据 |
|---|---|
| 裸探针出点 | /tmp/mid360_direct_probe_report.json：first_nonzero i=6 numElements=481 outputType=POINTCLOUD scanComplete=1 tsNs=1e8 |
| 探针读法 | LidarSensor(lidar, annotators=["generic-model-output"]) 创建时预注册 → get_data("generic-model-output") → data.numpy() → getModelOutputFromBuffer |
| native writer 死路 | IS6 注册表无 RtxLidarROS2PublishPointCloud/IsaacRtxLidarROS2PublishPointCloud/ROS2PublishPointCloud（WriterRegistryError），上棒已实证 |
| helper graph 死路 | 集成 transport=historical-helper：graph 里 ROS2RtxLidarHelper 成为第二 GMO 消费者，RELAY_DATA_READY=no + Warp CUDA 700（primdata.plugin Buffer.h:469，ARENA_READY 后 3s 即出） |
| Warp 700 只在集成复现 | 裸探针零错误 → 嫌疑锁定 graph 里 GMO 消费者初始化顺序 / tiled 相机 + helper 双消费者竞争 |
| rtx_sensors flag | tiled 相机 Camera.__init__ 置 /isaaclab/render/rtx_sensors=true 杀独立 lidar render product 的 GMO 流；_hold_rtx_sensor_render_mode 压制（保留） |

结论：集成里多出来的就是 `_create_direct_helper_lidar_graph`（ROS2RtxLidarHelper 节点 + ROS2Context + OnPlaybackTick 图）。探针链路（无 graph、单 GMO 消费者、轮询读数）已被实证出点且零 Warp 错误。

## 二、优选路线：direct_poll transport（裸探针移植）

**数据通路**（新）：
```
LidarSensor(annotators=["generic-model-output"])   ← 创建时预注册（探针同款）
  └ Kit 主循环每帧 process_pending()
      ├ creating 态: resource.ready()  ┐
      └ active   态: resource.health() ┘← 两态都调 _poll_and_publish（每帧一次）
          get_data("generic-model-output") → numpy → getModelOutputFromBuffer
          → numElements>0 且 signature(tsNs, numElements) 变化
          → 组装 PointCloud2（本进程 numpy 结构化数组）
          → rclpy publisher（sim 进程内，sensor_data QoS）→ /openflex/livox_frame/lidar
  └ livox_ros2_relay 子进程（不动）: /openflex/livox_frame/lidar → CustomMsg → /livox/lidar → DATA_READY
```

**绕开**：writer 注册表（native）、ROS2RtxLidarHelper graph（helper）、LidarEnableGate（无 graph 可关）。

## 三、改动清单

### A. sim_runtime/ros2/openflex_isaac_sim/openflex_isaac_sensors/openflex_isaac_sensors/mid360.py
1. `create_robot_mid360` transport 校验集合加 `"direct_poll"`：仅 `_create_lidar_render_product`（探针同款创建）+ resource_sink 记 `transport_used="direct_poll"`；不建 graph、不挂 writer。
2. 新增纯函数 `gmo_pointcloud2_buffer(parsed, frame_id)`：GMO 解析对象 → (packed_bytes, stamp_ns, numElements)。布局（对齐 relay 解析器 livox_ros2_relay.py 的字段需求 x/y/z 必需、intensity、line(channel/ring)、offset_time）：
   - fields: x F32@0, y F32@4, z F32@8, intensity F32@12, offset_time U32@16, line U8@20, pad 3B → point_step 24
   - per-point 字段防御式 getattr（timestamp→offset_time clip u32；channelId→line；intensity 缺省 0）
   - stamp 用 parsed.timestampNs
3. 同步拷贝到 ros2_pkgs/openflex_isaac_sim/...（两份历史上逐字节相同）。

### B. arena_benchmark/src/mrs_arena/sensors/mid360_lifecycle.py
1. `prepare_sensors()`：transport = `ARENA_MID360_TRANSPORT` env（默认 **direct_poll**；可回切 helper/native 复现对照）。
2. 新增 `_ensure_direct_publisher(resource)`：懒建 rclpy init + Node("arena_mid360_direct_publisher") + create_publisher(PointCloud2, source_topic, qos_profile_sensor_data)。探针已实证 IS6 venv 内 rclpy/sensor_msgs 可用。
3. 新增 `_poll_and_publish(resource) -> bool`：探针同款 get_data→decode→signature 去重→publish；异常打点 `ARENA_MID360_DIRECT_POLL_ERROR`。
4. `create_lidar().ready()`：direct_poll 分支 = 起 relay（共用 _start_livox_relay）→ ensure publisher → hold_rtx（保留）→ poll_and_publish → 返回 relay.has_data()（DATA_READY 即链路全通）。helper 分支原样保留。
5. `create_lidar().health()`：direct_poll = poll_and_publish 的成功与否（active 态每帧驱动点）；helper 原样。
6. `destroy()`：direct_poll 不关 gate（无 graph）；停 relay。publisher 生命周期归 shutdown()（destroy node + rclpy.shutdown try/except）。
7. 保留 `_report_gmo_state` / `_resubscribe_gmo`（stale 保底重订阅）。

### C. relay 侧
不动。契约输入= PointCloud2@/openflex/livox_frame/lidar（fields x/y/z/intensity/line/offset_time 均命中其解析器）。

## 四、验收（A6000，ground_truth + 雷达）

复用 /tmp/lidar_rebuild/run_stable_accept.sh（改链路 grep 关键字 transport=direct-poll）：
1. `ARENA_LIVOX_RELAY_DATA_READY` 出现 → `RELAY_DATA_READY=yes`
2. `ros2 topic hz /openflex/livox_frame/lidar` > 0（外部 shell 订测）
3. arena.log 中 `CUDA error 700` 计数 = 0（跑 2 分钟+）
4. 1 集 HDF5 落盘 `HDF5_VERDICT PASS`（lio_pose + camera_frames 在）
5. 传感器 API /v1/sensors lidar state=active

## 五、回退与对照
- `ARENA_MID360_TRANSPORT=helper` 一键回旧行为（对照复现 Warp 700 用）。
- 本改动不提交 git；GUI 数据模式三态、gate 机制、relay、采集编排零改动。

## 六、风险与卡点预案
- rclpy 在 sim venv publish 若与 relay DDS 不通（历史 bridge publish→relay 已验证同进程 rmw 互通，风险低）：先查 RMW_IMPLEMENTATION 与 ROS_LOCALHOST_ONLY env 快照，再查 QoS。
- GMO per-point timestamp 字段名若非 `timestamp`：offset_time 全 0（relay 端 timebase=frame stamp，仍出有效 CustomMsg，MVP 可接受）。
- tiled 相机并发拖慢 get_data（探针 v3 已见）：每帧一次 get_data 开销可控；stale 保底 RESUB 保留。
