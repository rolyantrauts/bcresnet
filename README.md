# Broadcasted Residual Learning for Efficient Keyword Spotting.
This repository contains the implementation for the paper presented in

**Byeonggeun Kim<sup>\*1</sup>, Simyung Chang<sup>\*1</sup>, Jinkyu Lee<sup>1</sup>, Dooyong Sung<sup>1</sup>, "Broadcasted Residual Learning for Efficient Keyword Spotting", Interspeech 2021.** [[ArXiv]](https://arxiv.org/abs/2106.04140)

*Equal contribution
<sup>1</sup> Qualcomm AI Research (Qualcomm AI Research is an initiative of Qualcomm Technologies, Inc.)

It contains the keyword spotting standard benchmark, [Google speech command datasets v1 and v2](https://arxiv.org/abs/1804.03209).

## Abstract
![! an image](resources/method.png)

We propose a broadcasted residual learning method for keyword spotting that achieves high accuracy with small model size and computational load, making it well-suited for use on resource-constrained devices such as mobile phones. The method involves configuring most of the residual functions as 1D temporal convolutions while still allowing 2D convolutions via a broadcasted-residual connection that expands the temporal output to the frequency-temporal dimension. This approach enables the network to effectively represent useful audio features with much less computation than conventional convolutional neural networks. We also introduce a novel network architecture called the Broadcasting-residual network (BC-ResNet) that leverages this broadcasted residual learning approach, and we describe how to scale the model according to the target device's resources. BC-ResNets achieve state-of-the-art results, achieving 98.0% and 98.7% top-1 accuracy on Google speech command datasets v1 and v2, respectively. Our approach consistently outperforms previous approaches while using fewer computations and parameters.

## Reference
If you find our work useful for your research, please cite the following:
```
@inproceedings{kim21l_interspeech,
  author={Byeonggeun Kim and Simyung Chang and Jinkyu Lee and Dooyong Sung},
  title={{Broadcasted Residual Learning for Efficient Keyword Spotting}},
  year=2021,
  booktitle={Proc. Interspeech 2021},
  pages={4538--4542},
  doi={10.21437/Interspeech.2021-383}
}
```

# BCResNet for Custom Keyword Spotting

This repository contains PyTorch implementation of **Broadcasted Residual Learning (BCResNet)** for Keyword Spotting (KWS).

It has been adapted from the [original Qualcomm research](https://github.com/Qualcomm-AI-research/bcresnet) to support:
* **Custom Datasets**: Train on your own "Wake Word" / "Unknown" / "Silence" folders.
* **Variable Audio Lengths**: Support for 1.0s, 1.5s, or any custom duration.
* **Edge Deployment**: Automatic export to **ONNX (Float32)** and **Quantized ONNX (Int8)** for Raspberry Pi Zero 2 and other edge devices.
Main.py has been retained but there is a dependency hell between
Opset 17: Supports STFT but crashes on the complex number types in your specific PyTorch version.
Opset 12: Handles types safely but does not support STFT

So internal log mel support doesn't seem possible and main2.py will have to be used instead and stream_test2.py
Main2.py just has a external audio front end for the model input

---

## 📂 Dataset Structure

The training script expects your data to be organized into `training`, `validation`, and `testing` folders. Inside each, create any number of subfolders for your specific classes (e.g., `wakeword`, `unknown`, `silence`).

**The script automatically detects class names from the folder names.**

```text
dataset/
├── training/
│   ├── wakeword/     # contains .wav files (e.g., 1000+ files)
│   ├── unknown/      # contains .wav files (random words/sounds)
│   └── silence/      # contains .wav files (background noise)
├── validation/
│   ├── wakeword/
│   ├── unknown/
│   └── silence/
└── testing/          # Optional (uses validation if missing)
    ├── wakeword/
    ├── unknown/
    └── silence/
 ```   
 Note: Audio files can be of any duration. They will be automatically padded or cropped to the --duration set in training arguments.

🚀 Quick Start
1. Install Requirements
```
pip install torch torchvision torchaudio soundfile onnx onnxruntime tqdm numpy
```
2. Standard Training (Recommended for Pi Zero 2)
This configuration uses 80 Mel bins and Sub-Spectral Normalization (SSN). This provides the highest accuracy and robustness against noise.
```
python main2.py --data_root ./dataset --duration 1.5 --n_mels 80
```
`main3.py` creates the streaming version of the model  

Understanding the Parameters:

--duration 1.5:

Increases the input window to 1.5 seconds. Essential for multi-word phrases (e.g., "Ok Google", "Hey Siri") that physically take longer to speak than a single word like "Stop".

--tau (Model Width Multiplier):

This controls the capacity of the model by scaling the number of channels (filters) in every layer.

Choices: [1, 1.5, 2, 3, 6, 8]

tau=1: The base model (Base Channels = 8). Fast, lightweight.

tau=1.5: Increases filters by 50% (Base Channels = 12).

tau=2: Doubles the filters (Base Channels = 16). Significant accuracy boost for complex words, but file size increases by ~4x.

tau=8: Massive model (Base Channels = 64). Likely too slow for Pi Zero 2, but very accurate.

Note: Unlike the original Qualcomm repo, this implementation enables SpecAugment for ALL tau values to ensure maximum robustness.  

3. Microcontroller Training (Tiny Models)
This configuration uses 40 Mel bins and disables SSN. Use this if you are targeting very low-power microcontrollers (e.g., ESP32, Cortex-M4) and need to save every bit of RAM/Compute.

```
python main2.py --data_root ./dataset --n_mels 40 --no_ssn
```
`main3.py` for streaming version  

⚙️ Input Parameters

--data_root	./dataset	Path to your dataset root folder.  
--duration	1.0	Target length of audio samples in seconds.  
--sample_rate	16000	Target sample rate (Hz).  
--n_mels	80	Number of Mel frequency bins (Vertical resolution).  
--no_ssn	False	Critical Flag: Disables Sub-Spectral Normalization (See below).  
--tau	1.0	Model width multiplier. Use 1.5 or 2.0 for larger/more accurate models.  
--batch_size	64	Training batch size.  
--device	auto	Force device: cuda, mps (Mac), cpu, or auto.  
--spec_prob Probability (0.0-1.0) of applying SpecAugment  
--epochs EPOCHS  
--warmup_epochs WARMUP_EPOCHS  
--lr Learning rate  
--patience end training run after X epochs with increase in accuracy  

⚠️ Important: The "Vanishing Frequency" Problem
If you wish to use 40 Mel Bins (common for microcontrollers), you MUST use the --no_ssn flag. Here is the deep technical reason why:

The BCResNet architecture aggressively downsamples the input in the Frequency (vertical) dimension. It reduces the height by a factor of 16x over the course of the network.

Sub-Spectral Normalization (SSN) works by splitting the frequency bands into 5 independent groups to normalize them separately. This creates a hard mathematical constraint: the feature map height at the deepest layer must be divisible by 5.

✅ Scenario A: 80 Mels (Standard)
Input: 80 pixels high

Layer 1: 40 pixels

Layer 2: 20 pixels

Layer 3: 10 pixels

Deepest Layer: 5 pixels

SSN Operation: Splits 5 pixels into 5 groups → 1 pixel per group. (Success)

❌ Scenario B: 40 Mels (Crash)
Input: 40 pixels high

Layer 1: 20 pixels

Layer 2: 10 pixels

Layer 3: 5 pixels

Deepest Layer: 2 pixels

SSN Operation: Tries to split 2 pixels into 5 groups → 0 pixels per group.

Result: RuntimeError: shape '[64, 200, 0, 151]' is invalid

Solution: If using n_mels=40, you must disable SSN so the network treats the frequency dimension as a single block rather than trying to split it.

📦 Output Artifacts
After training completes, the script exports the best model in three formats:

best_bcresnet.pth

The raw PyTorch weights. Use this to resume training or fine-tune later.

bcresnet_float32.onnx

Standard ONNX model. Perfect for Raspberry Pi, Linux, or Mac deployment.

bcresnet_int8.onnx  

Why ONNX f32 runs fast on ARM  
(The "NEON" Factor)
You might assume that 8-bit integers (int8) are always faster than 32-bit floats (f32).  
While true on dedicated NPU hardware (like the ethos-u on some microcontrollers), on general-purpose ARM chips (Cortex-A series found in Raspberry Pi, phones, etc.), f32 is incredibly optimized  
A. The NEON FPU is Native to FloatEvery modern ARM core has a NEON unit (Advanced SIMD).  
The Architecture: NEON registers are 128-bits wide.Vectorization: They can hold four 32-bit floats at once.Optimization: ARM has spent 15+ years optimizing the instruction set for single-precision float math.  
Operations like FMLA (Floating-point Fused Multiply-Add) can execute in a single clock cycle, doing 4 multiplications and 4 additions simultaneously.  
The Result: Standard math libraries (OpenBLAS, Eigen, XNNPACK) used by ONNX Runtime maximize this f32 throughput effortlessly.  
B. The int8 "Trap" on Older ARMIf you quantize a model to int8 on an ARM chip that lacks specific "Dot Product" instructions (like older Cortex-A53/A72 chips found in Raspberry Pi 3/4), the CPU actually struggles:No Native Instruction:   
It cannot do "multiply 8-bit by 8-bit and add to 32-bit accumulator" in one step.Up-casting: It has to spend cycles converting the int8 numbers back up to int16 or int32 just to do the math safely without overflowing.  
Shuffle Overhead: It spends time moving bits around registers.  
On many ARM CPUs, highly optimized f32 code beats unoptimized or non-hardware-accelerated int8 code because the FPU is a "first-class citizen" and the integer unit is working via workarounds.  
2. Why esp-dl Expects f32 Inputs  
Even though the ESP32-S3 (and esp-dl) runs the layers of the neural network in highly efficient int8 or int16, the interface expects f32.  
A. The Audio Feature Reality  
Your audio frontend outputs Log Mel Spectrograms.  
These are rarely nice round integers. They are values like -1.45, 0.003, 12.5.Spectrogram calculation involves cos, log, and norm—operations that require floating-point precision to maintain dynamic range.  
If you truncated these to integers before giving them to the library, you would lose massive amounts of quiet signal information (the "spectral whitespace" we discussed).  
B. The "Input Quantizer" Node  
The esp-dl framework (and models optimized via esp-ppq) includes a specialized input layer, effectively a bridge:  
Input: You provide float *data (High precision, dynamic range).Scale & Shift: The first layer of the model applies a pre-calculated scale factor.Example: Input (-2.0 to 2.0) $\times$ Scale (30) $\rightarrow$ Internal Int8 (-60 to 60).  
Processing: The rest of the network (Conv2D, etc.) runs in super-fast int8/int16 using the ESP32-S3's vector instructions.  
C. Calibration  
To know how to convert that float to an int, the esp-ppq tool runs a calibration step during export. It feeds thousands of f32 samples through the model to see the min/max values. It needs the input to be f32 so it can calculate that scale factor precisely.  
  
ARM (Raspberry Pi/Mobile)  
Why f32? Hardware Maturity: NEON FPUs are blazing fast and software libraries (BLAS) are fully optimized for f32 vectors.     
Why not int8? Instruction Gaps: Without sdot instructions, int8 can be slower due to casting overhead.  
ESP32-S3 (esp-dl)  
Precision Handoff: Audio features are naturally floats. The framework handles the "Float $\to$ Int" conversion internally to preserve accuracy.  
Usage: It does use int/int8 internally! It just hides it behind an f32 API for ease of use.

📝 License
This project is based on the original [BCResNet](https://github.com/Qualcomm-AI-research/bcresnet) implementation.   
  
🧠 [Understanding Hyperparameters: LR, Epochs, and Stability](https://github.com/rolyantrauts/bcresnet/discussions/2 )  
 
🎙️ [Designing the Perfect Wake Word: Phonetics & Strategy](https://github.com/rolyantrauts/bcresnet/discussions/3)  
  
🏠 [Domestic Room Impulse Response (RIR) Generator](https://github.com/rolyantrauts/bcresnet/blob/main/datasets/Room_Impulse_Response_(RIR)_Generator.md)  

🔊 [Calculate the embeddings on your own sound data](https://colab.research.google.com/drive/1E3CaPAqCai9P9QhJ3WYPNCVmrJU4lAhF)
