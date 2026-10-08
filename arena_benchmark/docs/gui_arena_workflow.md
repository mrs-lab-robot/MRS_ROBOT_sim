# GUI Arena 闭环

GUI 选择 `arena` 后端时，流程保持在 Arena Kit 进程内：

1. 在“任务配置”选择第一方 OpenFlex 任务或上游 Arena YAML。
2. 在同一面板设置固定/随机 seed、每次复位随机化、相机和采集回合数。
3. 在“仿真配置”填写 Arena 项目和 Python，启动 Arena Kit；真实 Pico 输入可额外启动 ROS 2 relay。
4. 在“数据采集”页通过 Capture API 开始、保存、丢弃 Episode，并将 HDF5 导出为 LeRobot。
5. 在“模型推理”页选择 checkpoint 和 `openflex_*` task，填写 seed range、episode 数和结果 JSON，GUI 会启动 `evaluate_arena_openflex.py`。

第一方任务位于 `configs/tasks` 和 `configs/scenes`，工厂位于
`mrs_arena.environments.openflex_task_registry`。资产和 OpenFlex contract 仍由
`sim_runtime` 的 canonical resolver 提供；上游 `third_party/IsaacLab-Arena` 只读。

首个搬箱任务：

```bash
bash scripts/run_native_isaac.sh \
  scripts/teleop_vr_openflex.py \
  --task-id openflex_dual_arm_box_transport \
  --scene-id openflex_box_transport \
  --seed 42 --episodes 10 --enable_cameras \
  --session-metadata outputs/datasets/session.json
```
