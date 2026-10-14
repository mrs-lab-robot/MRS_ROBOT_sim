# MID360 首帧数据——最终交付（45分钟收束版，已收口）

> 落盘说明：任务书指定路径 `.hermes/lidar-firstframe-final.md`（及预案名 final2）本会话被环境写保护（"protected agent-instruction file"），故同内容落到项目根目录。
> 写入时间：2026-10-14 ~01:22（本机时间）。本轮 4 次探针全部留档在 A6000:/tmp/mid360_direct_probe*.log + /tmp/mid360_direct_probe_report.json。

## 三、结论（先说答案）

**建议：搁置雷达，LIO 走真值+噪声路线先走通训练（用户已拍板的 MVP 主线，且已在跑：A6000 上 `teleop_vr_openflex.py --data-mode ground_truth --record-dir /tmp/mrs-gt-nolidar-r2/`，PID 233360，采集进行中）。雷达下个工作日再战。**

理由（本轮决定性实验 A/B 实测）：
- 独立链路（干净空舞台含 Ground+TargetBox×3+TargetWall、无 arena、无 tiled 相机、无 profile 属性写入）实测 **40/40 帧 GMO numElements=0，sizeB=352（纯 header），x_nonzero=0，tsNs 正常推进 0→3.9s** ——几何目标在跟前也一个点都不出。
- arena 探针（死前）症状同签名：`ARENA_MID360_GMO_DECODE numElements=0 outputType=OutputType.POINTCLOUD scanComplete=1 rows=0 cols=0 sizeB=352 x_nonzero=0`（tsNs 1.5s→1.6s 推进）。
- 两条链路同病 → 根因在 **Isaac Sim 6.0.0.1 的 `Lidar.create` → RTX 渲染 → GenericModelOutput 出数层**（config=`Example_Rotary`, tick 10Hz, accumulate_outputs=True, aux_output_level=FULL, supportMultiTickRate=true, srtx=off, multiGpu=off），不在 arena 集成层。arena 的 tiled 相机 `rtx_sensors` 互踩与 RESUB 循环是**第二层**问题（真实存在但不是首帧空点的根因）。

## 一、前置事实（浓缩自三轮调查 + 本轮远端复核，下个 agent 免重查）
- 探针仿真在 A6000 远端（ssh 别名 `6000PRO-WYH` / 100.100.2.30, user 1024201092WYH, host lenovo-ThinkStation-PX），探针日志：远端 /tmp/lidar_fix_a2.log（134KB, mtime 10-13 23:44）
- **本轮复核：探针已退出，PID 228423 不存在。** 死因链（日志行号）：
  1. `ARENA_MID360_RTX_SENSORS_HELD flag_false`（全日志 456 次，guard 反复把 `/isaaclab/render/rtx_sensors` 按回 false）
  2. L882-883：`TimeoutError: timed out waiting to create sensor lidar`（arena_sensor_api.py:191，create 请求进队列后 Kit 主线程 `process_pending()` 没能在超时窗内完成）
  3. `ARENA_STOP_REQUESTED` → `[rtx.hydra] Invalid USD RenderProduct Prim: ...rtx_sensor_8525150674015` + `HydraEngine::render failed to end the compute graph: error code 6`
  4. `SimulationContext cleared`（进程正常收尾）
- arena 侧已知机制（代码自证，`mid360_lifecycle.py`）：
  - tiled 相机 `Camera.__init__` 把 `/isaaclab/render/rtx_sensors` 置 true → RTX sensor 管线把雷达 render product 踢出管线 → GMO 流被杀（docstring 注明 verified A/B）→ guard 循环压制 + `_resubscribe_gmo` 重挂（RESUB #10→#140，日志 L388→L856，重挂对象都是同一 hydra texture 路径）
  - `create_robot_mid360` 的 native transport 在"请求时"才挂 writer，而 create 阶段就超时死了，根本没走到 writer
- 三轮调查坐实：无 DECODE 出点、无 DATA_READY（本轮修正：DECODE 其实有——但全是 numElements=0 的空帧）

## 二、本轮决定性实验（方案 A：独立实例化，不依赖探针仿真）

### 实验脚本
`/tmp/mid360_direct_probe.py`（本机同路径有副本），核心路径与 arena 读取完全同款以保证可比：
```python
import openflex_isaac_sensors.mid360 as m360
lidar = m360._create_lidar(stage, "/World/MID360/Lidar", tick_rate_hz=10.0)
m360._configure_profile(stage, lidar_path, "MID360_PERFORMANCE", ...)   # 默认跳过属性写入
rpp, sensor = m360._create_lidar_render_product(lidar, "livox_frame")   # LidarSensor(annotators=["generic-model-output"])
data, extra = sensor.get_data("generic-model-output")                    # 注意：返回元组
from isaacsim.sensors.experimental.rtx.generic_model_output import getModelOutputFromBuffer
parsed = getModelOutputFromBuffer(data.numpy())
# parsed.numElements / outputType / scanComplete / timestampNs / sizeInBytes / x
```
启动参数复刻 standalone worker 契约：`headless, manualModeEnabled=true, supportMultiTickRate=true, multiGpu=off, srtx=off, limit_cpu_threads=16`，venv python：`arena_benchmark/.venv/bin/python`。

### 实验实际输出（v4，干净复测，进程 235051，日志 /tmp/mid360_direct_probe_v4.log）
- 雷达创建成功：`lidar=/World/MID360/Lidar profile=MID360_PERFORMANCE rpp=/Render/OmniverseKit/HydraTextures/rtx_sensor_8361941258827`
- `sensor api: ['annotators','attach_annotators','attach_writer','authoring_object','detach_annotators','detach_writer','get_data','lidar','render_product']`
- **40/40 轮询全部**：`{"ok": true, "numElements": 0, "outputType": "OutputType.POINTCLOUD", "scanComplete": 0, "rows": 0, "cols": 0, "tsNs": 0→3.9s(每帧+100ms), "sizeB": 352, "x_nonzero": 0}`
- `GMO summary: ok_polls=40/40 max_numElements=0 first_nonzero=False`
- 舞台含几何：`mid360_empty_stage.usda` 内有 `Ground` Mesh + `TargetBox01/02/03` + `TargetWall`（77 行，L20-L66）——**非空场景，0 点不是物理正确**
- ROS 发布验证（best-effort）：`isaacsim.ros2.bridge` 等待注册后 ready=True，但三个 writer 名全部不在 IS6 replicator 注册表：`RtxLidarROS2PublishPointCloud` / `IsaacRtxLidarROS2PublishPointCloud` / `ROS2PublishPointCloud` 均 `WriterRegistryError` → **`mid360.py:_attach_native_pointcloud_writer` 在 IS6 上按现状是坏的**（writer 名是旧版 IS5 时代的）

### 前置失败迭代（给下个 agent 的坑位图）
- v1：`create_standalone_mid360(transport="direct")` 整链复刻 → 死在 `OmniGraphError: unrecognized type 'isaacsim.ros2.bridge.ROS2Context'`——`enable_extension` 是异步的，4 帧 update 不够，必须轮询 `og.get_node_type("isaacsim.ros2.bridge.ROS2Context")` 连续 3 帧非空才算 ready（v2 起修好，此法已验证可用）
- v2：`sensor.get_data()` 无参调用 → `TypeError: missing 1 required positional argument: 'annotator'`（正确姿势 `get_data("generic-model-output")`，返回元组）
- v3：在 GT 采集仿真（GPU 5.8G 占用）并发下 get_data 长时间无返回、日志停在被创建后；v4 清场复测全程畅通（boot ~110s，240 update + 40 poll ≈ 4s）——并发 GPU 采集会拖慢但不改变结论，v4 已与 GT 仿真同卡运行

### 三轮修复尝试清单（均未换来一条点云，免重查）
1. helper→native 切换（崩溃已修；native 依赖的 writer 名在 IS6 注册表不存在，见上）
2. profile 属性写入开关（`mid360.py` 注释已固化结论：IS6 上写 `omni:sensor:Core:{maxReturns,nearRangeM,farRangeM}` 任意一次即杀 GMO 流，numElements=0；默认已跳过；`OPENFLEX_MID360_LEGACY_PROFILE_ATTRS=1` 可还原旧行为做隔离实验）
3. render_mode 挖掘 + `--/rtx/hydra/supportMultiTickRate` 等 Kit 参数对比（arena=false vs worker=true，两条链路同空，排除）
4. RESUB 重订阅验证（#10→#140 次重挂同一 render product，GMO 依旧空帧，排除 annotator 脱挂假说）

## 四、建议后续路径

**采纳 MVP：搁置雷达，LIO 用真值+噪声先走通训练（主线，已开跑）。** 下个工作日重开雷达的打击清单（按性价比排序）：
1. 换 config：`OPENFLEX_ISAAC_LIDAR_CONFIG=MID360_APPROX`（→ multiScan100 + multiScan165/Profile01_20Hz_0p5deg 变体）再跑 v4 同款探针——当前 `Example_Rotary` 是唯一被验证空转的 config，变体 config 是最大嫌疑变量
2. 动 `Lidar.create` 参数矩阵：`accumulate_outputs=False`、`aux_output_level` 去掉 FULL、tick_rate 对齐 render 率；任一变量单独 A/B（探针脚本已就绪，改一行即可）
3. IS6 版本对策：查 Isaac Sim 6.0.0.1 release note 里 GenericModelOutput/RTX Lidar 的 breaking change；必要时拿 5.1 环境跑同探针定界"版本回退可用性"
4. 用 replicator 官方路径旁证：同一 render product 上挂 `rep.AnnotatorRegistry` 的 pointcloud 类 annotator 或 `rep.writers.BasicWriter`，验证是"渲染层空"还是"GMO 编码层空"
5. 修 `_attach_native_pointcloud_writer`：在 IS6 注册表里找新 writer 名（`rep.writers.WriterRegistry` 遍历打印一次即可拿到全名单），否则 native transport 上线即炸

## 附：本轮产物索引
- 探针脚本（本机）：/tmp/mid360_direct_probe.py（v3/v4 版，get_data+GMO 解码版）
- A6000 日志：/tmp/mid360_direct_probe.log（v1-v3）、/tmp/mid360_direct_probe_v4.log（决定性一轮）、/tmp/mid360_direct_probe_report.json（v4 完整 report）
- 死探针日志（arena 侧证据）：A6000:/tmp/lidar_fix_a2.log（134KB，勿删，下个 agent 的对照物）
- 本交付文件：MRS_ROBOT_sim/lidar-firstframe-final.md（.hermes/ 写保护，按预案落根目录）
