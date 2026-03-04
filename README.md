# YOLO PPE Helmet & Vest 实战项目

本项目包含完整的 PPE（安全帽/反光衣）目标检测工程流程：

- 训练：`best.pt` 产出与训练配置（`yaml` + 超参）
- 推理 Demo：Streamlit（上传图片/视频）+ OpenCV 摄像头实时检测
- 评估报告：mAP50-95、PR 曲线、混淆矩阵、FPS（CPU/GPU）
- 业务化产物：违规统计 `CSV` + 示例告警截图
- 工程化（可选）：Docker、ONNX/TensorRT 导出

---

## 1) 环境准备（不使用 base）

在 PowerShell 执行：

```powershell
cd "C:\Users\pengh\Desktop\YOLO 实战"
powershell -ExecutionPolicy Bypass -File .\scripts\setup_venv.ps1
.\.venv\Scripts\Activate.ps1
```

---

## 2) Kaggle 数据集下载

推荐数据集（任选其一，按你账号可访问情况）：

- `andrewmvd/hard-hat-detection`
- `andrewmvd/construction-site-safety-image-dataset`
- 其他包含 `Hardhat / NO-Hardhat / Safety Vest / NO-Safety Vest / Person` 标签的数据集

确保你本机已配置 `~/.kaggle/kaggle.json`（Windows 通常是 `C:\Users\你的用户名\.kaggle\kaggle.json`）。

下载示例：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\download_kaggle.ps1 -Dataset "andrewmvd/construction-site-safety-image-dataset"
```

下载后，将数据整理成如下结构：

```text
datasets/ppe-construction/
  images/train, images/val, images/test
  labels/train, labels/val, labels/test
```

并检查 `data/ppe_kaggle.yaml` 的 `path` 是否正确。

---

## 3) 训练（输出 best.pt）

训练配置在 `configs/train_config_hv.yaml`：

- 模型：`yolo11m.pt`
- 输入尺寸：`640`
- 轮次：`120`
- batch：`4`
- 优化器：`AdamW`
- 初始学习率：`0.001`

开始训练：

```powershell
python .\scripts\train.py --config .\configs\train_config_hv.yaml --name ppe_best_exp
```

训练结束后，模型会自动复制到：

- `artifacts/best.pt`

---

## 4) 推理 Demo

### 4.1 Streamlit 上传图片/视频

```powershell
streamlit run .\apps\streamlit_demo.py
```

浏览器中可上传图片或视频进行检测。

### 4.2 摄像头实时检测（OpenCV）

```powershell
python .\scripts\infer_camera.py --weights .\artifacts\best.pt --cam 0 --device cpu
```

GPU 示例：

```powershell
python .\scripts\infer_camera.py --weights .\artifacts\best.pt --cam 0 --device 0
```

---

## 5) 评估报告

执行：

```powershell
python .\scripts\validate_and_report.py --weights .\artifacts\best.pt --data .\data\ppe_kaggle.yaml --device cpu --fps_source .\outputs\demo_input.mp4
```

产物：

- 指标 JSON：`outputs/eval_report/metrics.json`（含 `mAP50-95`、`mAP50`、`precision`、`recall`、`fps`）
- PR 曲线/混淆矩阵：`runs/val/ppe_val/` 下自动生成图表

---

## 6) 业务化产物（违规统计）

执行：

```powershell
python .\scripts\violation_report.py --weights .\artifacts\best.pt --source .\outputs\demo_input.mp4 --device cpu
```

产物：

- `outputs/business/violation_stats.csv`
- `outputs/business/alerts/*.jpg`（违规告警截图）

---

## 7) 可选工程化

### 7.1 Docker

```powershell
docker build -t ppe-yolo:latest .
docker run --rm -p 8501:8501 ppe-yolo:latest
```

### 7.2 ONNX / TensorRT 导出

导出 ONNX：

```powershell
python .\scripts\export_models.py --weights .\artifacts\best.pt --onnx --device cpu
```

导出 TensorRT（需 NVIDIA 环境）：

```powershell
python .\scripts\export_models.py --weights .\artifacts\best.pt --engine --device 0
```

---

## 8) 版本记录建议

建议每次实验同步记录：

- Python / CUDA / torch / ultralytics 版本
- 数据集版本和划分比例
- 训练配置文件（`configs/train_config_hv.yaml`）与最终权重（`artifacts/best.pt`）
- 验证报告（`outputs/eval_report/metrics.json`）

可自动导出运行环境版本：

```powershell
python .\scripts\save_versions.py
```

输出文件：

- `artifacts/versions.json`
<img width="1024" height="683" alt="image" src="https://github.com/user-attachments/assets/5011dd3a-0958-415f-b2a0-f1f91f8f4414" />
