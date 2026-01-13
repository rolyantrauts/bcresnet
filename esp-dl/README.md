# Real-Time Wake Word & Speech Enhancement Pipeline

This repository focuses on high-accuracy, low-latency wake word detection and speech enhancement using State-of-the-Art (SotA) architectures like **BC-ResNet** and **H-GTCRN**. The core philosophy is **Signature Alignment**: ensuring your wake word engine is trained on audio that has been processed by the exact same enhancement stack used in production.

---

## 🏗️ Model Architectures

### 1. BC-ResNet (Wake Word Engine)
BC-ResNet is the current SotA for lightweight keyword spotting. It achieves >98% accuracy on GSC v2 with a fraction of the parameters of traditional CNNs by using Broadcasted Residual Learning.

* **ESP-DL Optimized:** Specifically designed to leverage the **ESP32-S3** vector instructions.
* **Ultra-Lightweight:** BC-ResNet-1 fits in ~10k parameters, ideal for 30ms streaming on microcontrollers.

### 2. H-GTCRN / GTCRN (Speech Enhancement)
**GTCRN** (Grouped Temporal Convolutional Recurrent Network) is a hybrid model for speech enhancement and noise suppression under low-SNR conditions.

* **Efficiency:** Features only 48.2 K parameters and 33.0 MMACs/s.
* **Hybrid Dual-Channel:** Utilizes spatial-spectral correlations for superior noise separation.
* **Streaming Support:** Recently updated with [causal streaming implementation](https://github.com/Xiaobin-Rong/gtcrn/commit/69f501149a8de82359272a1f665271f4903b5e34).

---

## 🚀 Deployment: From PyTorch to ESP32-S3

To deploy the **BC-ResNet** model on the **ESP-DL** framework, follow this optimized workflow:

### ⚙️ Conversion Steps
1.  **Export Float32 ONNX:** * Do **not** use generic `int8.onnx` files. 
    * Export from PyTorch using `opset_version=13` to ensure compatibility.
    * Example: `torch.onnx.export(model, dummy_input, "model.onnx", opset_version=13)`.
2.  **Quantization via ESP-PPQ:**
    * Install the tool: `pip install esp-ppq`.
    * Use the Float32 ONNX and calibration audio files to generate the `.espdl` file.
3.  **External Feature Extraction:** * Log-Mels should be calculated externally (see `main2.py`) to match the [ESP Audio Front-End](https://github.com/espressif/esp-dl/tree/master/esp-dl/audio).

### 📋 ESP-DL Operator Support Matrix
| Layer Type | ONNX Op | ESP-DL Status |
| :--- | :--- | :--- |
| **Conv2D** | `Conv` | ✅ Optimized for S3 Vector Instructions |
| **BatchNorm** | `BatchNormalization` | ✅ Fused during conversion |
| **Residual Add** | `Add` | ✅ Supported |
| **Pooling** | `GlobalAveragePool` | ✅ Supported |
| **Sub-Spectral Norm** | `Reshape` / `Trans` | ✅ Supported via tensor manipulation |

---

## 🛠️ Training & Dataset Strategy: "Signature Alignment"

Speech enhancement models leave a "processing signature." If your wake word engine is trained on clean audio but deployed behind a noise suppressor like **DTLN** or **GTCRN**, accuracy will suffer.

### 1. The Matched Pipeline
* **Step 1:** Select your enhancement model (e.g., [PiDTLN](https://github.com/SaneBow/PiDTLN) for PiZero2 or GTCRN for ESP32).
* **Step 2:** Run your entire wake word dataset through that model's processing chain.
* **Step 3:** Train the wake word model (BC-ResNet) on this "processed" audio.

### 2. Training Parameters
* **Learning Rate:** Use **0.005** with a safety fallback to **0.001** if NaN loss is detected.
* **Segment Length:** 1.0s to 3.0s segments at 16kHz.
* **Spatial Simulation:** For dual-channel systems, use `pyroomacoustics` to simulate mic distances > 75mm, as ML can handle the spatial aliasing that breaks traditional DSP.

---

## 📂 Source References
* [GTCRN (Xiaobin-Rong)](https://github.com/Xiaobin-Rong/gtcrn)
* [BC-ResNet (rolyantrauts)](https://github.com/rolyantrauts/bcresnet)
* [PiDTLN (SaneBow)](https://github.com/SaneBow/PiDTLN)
* [Espressif ESP-DL](https://github.com/espressif/esp-dl)
