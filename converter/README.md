# 模型格式转换工具（converter）

在 `converter` 目录下提供 **TFLite → ONNX** 与 **TFLite → MNN** 两套脚本。

## 目录结构

```
converter/
├── README.md
├── setup_all.sh                 # 一次性安装 ONNX + MNN 环境
├── setup_env.sh                 # ONNX：Python venv + pip 依赖
├── setup_mnn_env.sh             # MNN：编译 MNNConvert
├── requirements.txt             # ONNX 流水线 Python 依赖
├── convert_tflite_to_onnx.sh
├── convert_tflite_to_mnn.sh
├── mnn_runtime.sh               # MNNConvert 路径与 LD_LIBRARY_PATH
├── tools/mnn/                   # 预置 MNNConvert 及依赖 .so（优先使用）
│   ├── MNNConvert
│   ├── libMNN.so
│   ├── express/libMNN_Express.so
│   └── tools/converter/libMNNConvertDeps.so
└── output/                      # 转换产物（可选）
```

---

## 快速开始（全部环境）

```bash
cd /media/zhuqingquan/project/project/ModelFactory/converter
chmod +x *.sh
./setup_all.sh
```

---

## TFLite → ONNX

使用 [tf2onnx](https://github.com/onnx/tensorflow-onnx)。

### 安装

```bash
./setup_env.sh
```

**注意：** 项目目录在 **NTFS**（`fuseblk`）上时，请勿在 `converter/.venv` 创建虚拟环境，易报 `Errno 28`。请使用 ext4 路径：

```bash
VENV_DIR=$HOME/.venvs/mf_converter ./setup_env.sh
VENV_DIR=$HOME/.venvs/mf_converter ./convert_tflite_to_onnx.sh ...
```

### 转换

```bash
./convert_tflite_to_onnx.sh ../facedetect/model/blaze_face_short_range.tflite output/blaze_face_short_range.onnx
```

---

## TFLite → MNN

使用阿里巴巴 [MNN](https://github.com/alibaba/MNN) 官方的 **MNNConvert** 工具。

### 安装

**默认无需编译**：`tools/mnn/` 已包含预构建的 `MNNConvert` 及 `libMNN.so` 等依赖，可直接转换。

仅当预置工具不可用（换机器、架构不匹配等）时再执行：

```bash
./setup_mnn_env.sh
```

脚本逻辑：

1. 检测 `tools/mnn/` 内二进制是否可运行 → **可用则跳过构建**
2. 否则克隆 MNN 到 `~/.cache/mf_mnn/MNN`，在 `~/.cache/mf_mnn/build` 编译
3. 将编译结果复制到 `tools/mnn/` 供后续直接使用

依赖（仅编译时需要）：`git`、`cmake`、`g++`、`protobuf`。若编译失败可安装：

```bash
sudo apt install -y git cmake g++ libprotobuf-dev protobuf-compiler
```

### 转换

```bash
./convert_tflite_to_mnn.sh ../facedetect/model/blaze_face_short_range.tflite output/blaze_face_short_range.mnn
```

`convert_tflite_to_mnn.sh` 会自动设置 `LD_LIBRARY_PATH` 加载 `tools/mnn` 下的 `.so`。

若直接转 TFLite 失败，可走 ONNX 中转：

```bash
CONVERT_VIA_ONNX=1 ./convert_tflite_to_mnn.sh input.tflite output.mnn
```

（需先完成 `setup_env.sh`）

---

## facedetect BlazeFace 示例

| 模型 | ONNX 输入/输出 | MNN 输入/输出 |
|------|----------------|---------------|
| blaze_face_short_range | `input` [1,128,128,3] → `regressors`, `classificators` | 同左 |
| blaze_face_full_range | `input` [1,192,192,3] → `reshaped_regressor_face_4`, `reshaped_classifier_face_4` | 同左 |

示例：

```bash
./convert_tflite_to_onnx.sh ../facedetect/model/blaze_face_short_range.tflite output/blaze_face_short_range.onnx
./convert_tflite_to_mnn.sh  ../facedetect/model/blaze_face_short_range.tflite output/blaze_face_short_range.mnn
```

项目内已有 MNN 推理示例：`facedetect/test/testVersionRFB_mnn.cpp`（UltraFace + `.mnn`）。

---

## 说明

| 场景 | 建议 |
|------|------|
| 有 SavedModel / PyTorch 源模型 | 优先从源格式直接转 ONNX/MNN，比经 TFLite 更稳 |
| 量化 TFLite、自定义算子 | ONNX 可试 [tflite2onnx](https://github.com/zhenhuaw-me/tflite2onnx)；MNN 可试 `CONVERT_VIA_ONNX=1` |
| Python venv 安装失败 | venv 放在 `$HOME` 或 `/tmp`（ext4），不要放在 NTFS 项目盘 |
| MNN 编译慢 | 优先用 `tools/mnn` 预置包；仅失败时再编译，`MNN_BUILD_DIR` 放 ext4 |

转换后建议用 ONNX Runtime / MNN 推理各跑一次，核对输出 shape 与数值。
