#include <Arduino.h>
#include "esp_heap_caps.h"

// Define standard audio format
using SampleType = int16_t; // 16-bit audio
const int SAMPLE_RATE = 16000;

class AudioRingBuffer {
private:
    SampleType* buffer = nullptr;
    size_t size = 0;
    size_t mask = 0;
    volatile size_t writeHead = 0;
    bool inPSRAM = false;

public:
    // Constructor: Allocates buffer in PSRAM
    // sizeInBytes: Recommend 65536 (64KB) or 131072 (128KB)
    AudioRingBuffer(size_t sizeInBytes) {
        // 1. Force Power of 2 for speed optimization
        // If user asks for 60000, we bump it to 65536
        size_t samples = sizeInBytes / sizeof(SampleType);
        size = 1;
        while (size < samples) size <<= 1; 
        
        mask = size - 1; // e.g., 65535 (0xFFFF)

        // 2. Allocate in PSRAM (SPIRAM)
        // MALLOC_CAP_SPIRAM forces allocation in external RAM
        // MALLOC_CAP_8BIT ensures byte-accessible memory
        buffer = (SampleType*)heap_caps_malloc(size * sizeof(SampleType), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

        if (buffer == nullptr) {
            Serial.println("Error: Failed to allocate PSRAM buffer! Fallback to internal RAM.");
            // Fallback to internal RAM if PSRAM fails (unlikely on S3)
            buffer = (SampleType*)heap_caps_malloc(size * sizeof(SampleType), MALLOC_CAP_INTERNAL);
            inPSRAM = false;
        } else {
            inPSRAM = true;
            memset(buffer, 0, size * sizeof(SampleType));
        }
        
        Serial.printf("Audio Buffer: %u samples (%u bytes) allocated in %s\n", 
                      size, size * sizeof(SampleType), inPSRAM ? "PSRAM" : "Internal RAM");
    }

    ~AudioRingBuffer() {
        if (buffer) free(buffer);
    }

    // --- Fast Write ---
    // Called by I2S Interrupt or Loop
    void write(SampleType sample) {
        if (!buffer) return;
        
        buffer[writeHead] = sample;
        // Optimized wrapping: (i + 1) & mask is faster than (i + 1) % size
        writeHead = (writeHead + 1) & mask;
    }

    // --- Fast Write Block ---
    void write(const SampleType* data, size_t len) {
        for (size_t i = 0; i < len; i++) {
            write(data[i]);
        }
    }

    // --- Event Extraction ---
    // Extracts the last 'durationSeconds' of audio into a linear buffer
    // Returns number of samples actually copied
    size_t extractLast(float durationSeconds, SampleType* destination) {
        if (!buffer) return 0;

        size_t samplesToRead = (size_t)(durationSeconds * SAMPLE_RATE);
        if (samplesToRead > size) samplesToRead = size; // Cap at buffer limit

        // Calculate start index (going backwards, handling wrap)
        // We add 'size' before subtracting to prevent negative underflow
        size_t readHead = (writeHead + size - samplesToRead) & mask;

        // Copy logic: The data might wrap around the end of the array
        // Case A: [ ... Start ...... End ... ] (Linear)
        // Case B: [ ... End ...... Start ... ] (Wrapped)
        
        size_t samplesToEnd = size - readHead;

        if (samplesToRead <= samplesToEnd) {
            // Case A: Contiguous copy
            memcpy(destination, &buffer[readHead], samplesToRead * sizeof(SampleType));
        } else {
            // Case B: Split copy
            // 1. Copy from ReadHead to End of Buffer
            memcpy(destination, &buffer[readHead], samplesToEnd * sizeof(SampleType));
            // 2. Copy remainder from Start of Buffer
            memcpy(&destination[samplesToEnd], &buffer[0], (samplesToRead - samplesToEnd) * sizeof(SampleType));
        }

        return samplesToRead;
    }
    
    // Debug: Check if memory is healthy
    bool isReady() { return buffer != nullptr; }
};
