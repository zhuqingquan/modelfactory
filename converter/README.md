# TFLite → ONNX 转换

推荐使用 [tf2onnx](https://github.com/onnx/tensorflow-onnx)（微软维护，官方支持 `.tflite`）。

## 目录结构

```
converter/
├── README.md
├── requirements.txt
├── setup_env.sh              # 初次运行：安装 Python 依赖
├── convert_tflite_to_onnx.sh # 执行 TFLite → ONNX 转换
└── output/                   # 转换产物（可选）
```

## 初次运行：安装环境

在 `converter` 目录下执行：

```bash
cd /media/zhuqingquan/project/project/ModelFactory/converter
chmod +x setup_env.sh convert_tflite_to_onnx.sh
./setup_env.sh
```

脚本会创建 `.venv` 并安装 `requirements.txt` 中的依赖（`tensorflow`、`tf2onnx`、`onnx`）。

若项目盘创建 venv 失败，可指定其他路径：

```bash
VENV_DIR=/tmp/mf_converter_venv ./setup_env.sh
```

然后转换时使用：`PYTHON=/tmp/mf_converter_venv/bin/python ./convert_tflite_to_onnx.sh ...`

## 转换模型

```bash
./convert_tflite_to_onnx.sh ../facedetect/model/blaze_face_short_range.tflite output/blaze_face_short_range.onnx
```

或直接使用 venv 中的 Python：

```bash
.venv/bin/python -m tf2onnx.convert \
  --opset 16 \
  --tflite /path/to/model.tflite \
  --output /path/to/model.onnx
```

## facedetect 模型示例

`facedetect/model` 下两个 BlazeFace TFLite 模型可参考如下 IO（转换后 ONNX）：

| 模型 | 输入 | 输出 |
|------|------|------|
| blaze_face_short_range | `input` [1,128,128,3] float32 | `regressors` [1,896,16], `classificators` [1,896,1] |
| blaze_face_full_range | `input` [1,192,192,3] float32 | `reshaped_regressor_face_4`, `reshaped_classifier_face_4` |

## 说明

- 若仍有 **SavedModel / Keras / frozen .pb**，优先直接从源格式转 ONNX，通常比「先 TFLite 再 ONNX」更稳。
- 量化 TFLite、含自定义算子的模型可能转换失败，可尝试 [tflite2onnx](https://github.com/zhenhuaw-me/tflite2onnx) 或从训练框架重新导出。
- 转换后建议用 ONNX Runtime 或现有推理代码做一次数值对比验证。
