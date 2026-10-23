# MoRe 视频逐帧抽取、批量推理与 Motion Mask 可视化

## 1. 文件放置

将以下两个脚本复制到服务器的同一目录：

```text
/workspace/MoRe/code/MoRe/tools/more_pipeline/
├── process_one_video.py
└── process_batch_videos.py
```

使用现有的 `(more)` Conda 环境，要求 `ffmpeg`、`ffprobe` 可执行，且 MoRe 推理脚本、权重、配置、可视化脚本位于现有路径。

```bash
conda activate more
cd /workspace/MoRe/code/MoRe
which ffmpeg ffprobe
python --version
```

## 2. 正式批量前的两级验证

首先，对 `raw-video/` 下所有 `.mp4` 进行静态检查：核验所需文件、脚本语法、视频帧数、分辨率和平均帧率；此步**不会执行 GPU 推理**。

```bash
python tools/more_pipeline/process_batch_videos.py --preflight-only
```

然后，每个视频只取**前 32 帧**分别进行抽帧→MoRe 推理→mask 可视化的端到端检查。每条序列会检查输入帧、motion mask、PLY、深度和相机矩阵及可视化图片的完整数量；校验后删除成功的 smoke 临时数据，但保留日志。

```bash
python tools/more_pipeline/process_batch_videos.py --smoke-only --smoke-frames 32
```

> 注意：这两步只验证功能和短序列可运行性，不能保证长视频在 `more_full.pt` 全注意力模型下具有足够显存。脚本不会自动截断或分块正式推理。

## 3. 正式批量运行

```bash
python tools/more_pipeline/process_batch_videos.py
```

这是**三阶段控制流程**：先检查全部视频，再针对全部视频做短序列 smoke 测试，两步全通过后，再依次完整处理每一个原始视频。

如果已单独通过 smoke 检查，避免再次执行：

```bash
python tools/more_pipeline/process_batch_videos.py --skip-smoke
```

如果上次批量运行中途失败，修复故障后继续：

```bash
python tools/more_pipeline/process_batch_videos.py --resume --skip-smoke
```

`--resume` 仅跳过同时满足**已存在目录、期望帧数完整、输入与参数状态一致**的步骤。不会覆盖已有结果；如果已有输出不符合当前状态，脚本将失败并要求人工检查。

## 4. 单视频运行

用 `30825.mp4` 举例：

```bash
# 只做依赖和帧数检查
python tools/more_pipeline/process_one_video.py \
  --video /workspace/MoRe/data-tiny/raw-video/30825.mp4 --mode check

# 先试运行前 32 帧的完整三阶段流程
python tools/more_pipeline/process_one_video.py \
  --video /workspace/MoRe/data-tiny/raw-video/30825.mp4 \
  --mode smoke --smoke-frames 32

# 对全视频逐帧抽取并推理，再叠加绿色 motion mask
python tools/more_pipeline/process_one_video.py \
  --video /workspace/MoRe/data-tiny/raw-video/30825.mp4 --mode run

# 续跑（仅跳过校验通过的步骤）
python tools/more_pipeline/process_one_video.py \
  --video /workspace/MoRe/data-tiny/raw-video/30825.mp4 --mode run --resume
```

## 5. 路径与耗时

对每个 `name.mp4`：

| 输出 | 目录 |
| --- | --- |
| 无损 PNG 帧 | `/workspace/MoRe/data-tiny/name/frame_00000000.png` 等 |
| MoRe 输出 | `/workspace/MoRe/results/name_full_glbfix/` |
| Motion mask 叠加图 | `/workspace/MoRe/results/name_motion_overlay/` |
| 单次运行日志和 JSON | `/workspace/MoRe/results/pipeline_logs/name/<运行编号>/` |
| 最近批次记录 | `/workspace/MoRe/results/pipeline_logs/batch_<运行编号>/batch_summary.csv` |
| 每个序列的重跑校验状态 | `/workspace/MoRe/results/pipeline_state/name.json` |

CSV 按序列记录 `frames`、`fps`、`extract_s`、`inference_s`、`visualize_s`、`total_s`、三个 `*_skipped`、`status` 和 `error`。单项耗时包括该步骤的输出数量核验及磁盘 I/O；跳过的步骤耗时近似 0（不代表首次执行耗时）。首次成功执行的时长可从对应历史批次 CSV 或单次运行 `summary.json` 查找。

## 6. 可靠性说明

- 使用 FFmpeg `-fps_mode passthrough`，**不指定 `-r` 或 `fps` 滤镜**，避免丢帧、插帧；原始流的平均帧率作为精确有理数字符串写入 JSON/CSV。PNG 本身不携带播放帧率。**如原始视频是可变帧率（VFR），仅凭图像序列和平均帧率无法还原逐帧原始时间戳。**
- 帧文件用 `frame_%08d.png` 顺序编号，推理使用 MoRe 既有的排序机制；mask 保持 MoRe 原有的 `mask_%05d.png` 编号。
- 严格检查抽帧总数、编号连续性，以及模型导出的 `masks/*.png`、`frame_*.ply`、`depth/*.npy`、`extrinsic/*.npy`、`intrinsic/*.npy` 数量（均应一帧一份）。若本地 `inference.py` 的导出规则已经发生实质变化，应相应调整 `verify_inference()`，不能忽略失败。
- 每阶段先在同父目录写 `.partial.*` 临时目录，数量核对完成后才以目录重命名发布正式结果；失败时保留该临时目录以供排查，**不会覆盖已存在的输出目录**。
- 同一序列使用 Linux 文件锁，避免两个进程同时处理并污染输出。模型推理及可视化通过当前 Python 解释器运行，避免 Conda 环境混用。
- 默认批次逐视频串行执行，不会同时加载六份模型到 GPU。
- 全量 PNG、PLY、深度矩阵可能产生远超 MP4 大小的磁盘占用；运行前检查 `df -h /workspace/MoRe`。模型按完整序列进行计算，长序列可能显存不足；**脚本不会悄悄跳帧、缩帧数或分段处理**。

## 7. 模拟测试说明

两个脚本已经通过本地 Python 语法检查，并使用两个不同帧率的模拟 MP4 和占位推理/可视化程序验证：预检查、smoke、完整批次、断点重跑以及模型输出不完整时的报错退出。**这些测试不是在你的服务器上执行真实 MoRe GPU 推理**，实际运行前应完成第 2 节的端到端验证。
