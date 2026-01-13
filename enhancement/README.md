# Speech Enhancement & Wake Word Optimization

Optimizing wake word accuracy requires a holistic approach where the training environment matches the deployment environment. When a wake word engine is fed audio processed by speech enhancement, blind source separation, or beamforming, it encounters a specific "processing signature." To maximize performance, it is critical to **process your training dataset through the same enhancement pipeline used in production.**

---

## 🚀 Recommended Architectures

### 1. DTLN (Dual-Signal Transformation LSTM Network)
**Project Link:** [PiDTLN (SaneBow)](https://github.com/SaneBow/PiDTLN) | **Core Source:** [DTLN (breizhn)](https://github.com/breizhn/DTLN)

DTLN represents a paradigm shift from older methods like RNNNoise. It is lightweight, effective, and optimized for real-time edge usage.

* **Capabilities:** Real-time Noise Suppression (NS) and Acoustic Echo Cancellation (AEC).
* **Hardware Target:** Runs efficiently on **Raspberry Pi Zero 2**.
* **Key Advantage:** Being fully open-source, you can retrain DTLN with your specific wake word data.
* **Optimization Tip:** Augment your wake word dataset by running your clean samples through the DTLN processing chain. This "teaches" the wake word model to ignore the specific artifacts or spectral shifts introduced by the NS/AEC stages.

### 2. GTCRN (Grouped Temporal Convolutional Recurrent Network)
**Project Link:** [GTCRN (Xiaobin-Rong)](https://github.com/Xiaobin-Rong/gtcrn)

GTCRN is an ultra-lightweight model designed for systems with severe computational constraints. It significantly outperforms RNNNoise while maintaining a similar footprint.

| Metric | Specification |
| :--- | :--- |
| **Parameters** | 48.2 K |
| **Computational Burden** | 33.0 MMACs per second |
| **Performance** | Surpasses RNNNoise; competitive with much larger models. |

* **Streaming Support:** Recent updates have introduced a [streaming implementation](https://github.com/Xiaobin-Rong/gtcrn/commit/69f501149a8de82359272a1f665271f4903b5e34), making it viable for live audio pipelines.
* **Hardware Target:** Ideally suited for high-end microcontrollers (like **ESP32-S3**) and single-board computers.

---

## 🛠 Dataset Construction & Training Strategy

To achieve high-accuracy wake word detection under low SNR (Signal-to-Noise Ratio) conditions, follow this "Matched Pipeline" strategy:

1.  **Matched Pre-processing:** Whatever enhancement model you choose (DTLN or GTCRN), run your entire training corpus through it.
2.  **Signature Alignment:** Wake words processed by these models carry a unique "signature." If the model is trained on "dry" audio but deployed behind an NS filter, accuracy will drop. Training on "processed" audio closes this gap.
3.  **Low-Latency Streaming:** Ensure you are using the streaming variants of these models to keep the system latency low enough for a natural user experience (aiming for < 200ms total trigger latency).

---

> **Note:** For ESP32-S3 deployments, GTCRN is the preferred choice due to its ultra-low parameter count and MMAC requirements, fitting well within the constraints of the ESP-DL framework.
