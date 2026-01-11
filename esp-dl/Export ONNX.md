Step 1: Export ONNX
Ensure you export your model in main3.py with dynamic axes or a fixed shape that matches MODEL_INPUT_FRAMES.

Python

# Python export snippet
dummy_input = torch.randn(1, 1, 40, 64) # Example 64 frame context
torch.onnx.export(model, dummy_input, "bcresnet.onnx", ...)
Step 2: Quantize & Generate Code (ESP-PPQ)
Use the Espressif PPQ quantization tool to convert the ONNX to ESP-DL compatible C++ code.

```
python -m esp_ppq.quantize \
    --model bcresnet.onnx \
    --input_shape 1,1,40,64 \
    --target_chip esp32s3 \
    --output_path ./codegen
```
This will generate model_define.hpp (the class wrapper) and dl_tool.hpp.

Step 3: Project Structure
Your ESP-IDF project should look like this:

/my_project
  /components
     /esp-dl      <-- Clone from github.com/espressif/esp-dl
  /main
     AudioFrontEnd.cpp (or inside main.cpp)
     main.cpp
     model_define.hpp (Generated)
     dl_tool.hpp      (Generated)
     ... weights files ...
  CMakeLists.txt
3. Optimization Note (ESP-DSP)
The code above uses a generic fft function. For real-time 20ms performance on ESP32, you must use the hardware-accelerated FFT.

Add esp-dsp to your components.

In main.cpp:

C++
```
#include "dsps_fft2r.h"
// In initialization
dsps_fft2r_init_fc32(NULL, CONFIG_DSP_MAX_FFT_SIZE);
// In process loop
dsps_fft2r_fc32(spectrum_buffer, N_FFT);
dsps_bit_rev_fc32(spectrum_buffer, N_FFT);
```
This reduces the FFT cost from ~3ms to <0.5ms on an ESP32-S3.
