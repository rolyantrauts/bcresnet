#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <complex>
#include <cstring>

// Constants for BCResNet 20ms streaming (Adjust based on main3.py exact values)
constexpr int SAMPLE_RATE = 16000;
constexpr int FRAME_SIZE_MS = 32;   // Window size (usually larger than hop)
constexpr int HOP_SIZE_MS = 20;     // Stride (20ms for streaming)
constexpr int N_MELS = 40;          // BCResNet standard
constexpr int N_FFT = 512;          // Nearest power of 2 for 32ms (512 >= 16 * 32)
constexpr float F_MIN = 20.0f;
constexpr float F_MAX = 8000.0f;    // Nyquist

// Helper: PI
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class AudioFrontEnd {
private:
    int sample_rate_;
    int n_fft_;
    int n_mels_;
    int hop_length_;
    int win_length_;
    
    std::vector<float> window_;
    std::vector<std::vector<float>> mel_filters_;
    std::vector<int16_t> input_buffer_; // For streaming overlap
    
    // Internal FFT implementation (Simple Cooley-Tukey for dependency-free code)
    // In production, link against fftw3 or esp-dsp
    void fft(std::vector<std::complex<float>>& x) {
        size_t n = x.size();
        if (n <= 1) return;

        std::vector<std::complex<float>> even(n / 2), odd(n / 2);
        for (size_t i = 0; i < n / 2; ++i) {
            even[i] = x[2 * i];
            odd[i] = x[2 * i + 1];
        }

        fft(even);
        fft(odd);

        for (size_t i = 0; i < n / 2; ++i) {
            std::complex<float> t = std::polar(1.0f, (float)(-2 * M_PI * i / n)) * odd[i];
            x[i] = even[i] + t;
            x[i + n / 2] = even[i] - t;
        }
    }

    void create_hann_window() {
        window_.resize(win_length_);
        for (int i = 0; i < win_length_; ++i) {
            window_[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (win_length_ - 1)));
        }
    }

    // Create Mel Filterbank (Slaney-like implementation common in librosa/esp-dl)
    void create_mel_filters() {
        mel_filters_.resize(n_mels_, std::vector<float>(n_fft_ / 2 + 1, 0.0f));
        
        auto hz_to_mel = [](float hz) { return 1127.0f * std::log(1.0f + hz / 700.0f); };
        auto mel_to_hz = [](float mel) { return 700.0f * (std::exp(mel / 1127.0f) - 1.0f); };
        
        float mel_min = hz_to_mel(F_MIN);
        float mel_max = hz_to_mel(F_MAX);
        
        std::vector<float> mel_points(n_mels_ + 2);
        std::vector<int> bin_points(n_mels_ + 2);
        
        for (int i = 0; i < n_mels_ + 2; ++i) {
            mel_points[i] = mel_min + i * (mel_max - mel_min) / (n_mels_ + 1);
            float freq = mel_to_hz(mel_points[i]);
            bin_points[i] = std::floor((n_fft_ + 1) * freq / sample_rate_);
        }
        
        for (int i = 0; i < n_mels_; ++i) {
            for (int j = bin_points[i]; j < bin_points[i+1]; ++j) {
                mel_filters_[i][j] = (float)(j - bin_points[i]) / (bin_points[i+1] - bin_points[i]);
            }
            for (int j = bin_points[i+1]; j < bin_points[i+2]; ++j) {
                mel_filters_[i][j] = (float)(bin_points[i+2] - j) / (bin_points[i+2] - bin_points[i+1]);
            }
        }
    }

public:
    AudioFrontEnd() : 
        sample_rate_(SAMPLE_RATE),
        n_fft_(N_FFT),
        n_mels_(N_MELS),
        hop_length_(SAMPLE_RATE * HOP_SIZE_MS / 1000), // 320 samples
        win_length_(SAMPLE_RATE * FRAME_SIZE_MS / 1000) // 512 samples
    {
        create_hann_window();
        create_mel_filters();
        input_buffer_.reserve(win_length_);
    }

    // Process a chunk of audio (e.g., 20ms / 320 samples)
    // Returns a vector of Mel features for this frame (size N_MELS)
    std::vector<float> process_frame(const int16_t* data, size_t len) {
        // Append new data to buffer
        input_buffer_.insert(input_buffer_.end(), data, data + len);

        // If we don't have enough data for a full window, return empty or handle buffering
        // For strict streaming 20ms stride with 32ms window, we need 'win_length' samples.
        // We typically keep the last (win_length - hop_length) samples for the next overlap.
        
        if (input_buffer_.size() < (size_t)win_length_) {
            return {}; // Not enough data yet (cold start)
        }

        // 1. Prepare Frame (Apply Window)
        std::vector<std::complex<float>> spectrum(n_fft_, 0.0f);
        
        // We take the *last* win_length samples if we are streaming continuously,
        // or just the current buffer if synchronized. 
        // Assuming 'process_frame' is called with 'hop_length' samples:
        size_t start_idx = input_buffer_.size() - win_length_; 
        
        for (int i = 0; i < win_length_; ++i) {
            float sample = static_cast<float>(input_buffer_[start_idx + i]) / 32768.0f; // Normalize int16
            spectrum[i] = sample * window_[i];
        }

        // 2. FFT
        fft(spectrum);

        // 3. Power Spectrum (Magnitude Squared)
        std::vector<float> power_spec(n_fft_ / 2 + 1);
        for (int i = 0; i < n_fft_ / 2 + 1; ++i) {
            float real = spectrum[i].real();
            float imag = spectrum[i].imag();
            power_spec[i] = (real * real + imag * imag) / n_fft_; // Standard normalization
        }

        // 4. Mel Filterbank
        std::vector<float> mel_energies(n_mels_, 0.0f);
        for (int i = 0; i < n_mels_; ++i) {
            for (int j = 0; j < n_fft_ / 2 + 1; ++j) {
                mel_energies[i] += power_spec[j] * mel_filters_[i][j];
            }
        }

        // 5. Logarithm (Log10 + epsilon)
        // esp-dl often uses a specific scaling or natural log, but log10 is common for BCResNet
        constexpr float epsilon = 1e-6f;
        for (float& energy : mel_energies) {
            energy = 10.0f * std::log10(energy + epsilon); 
            // Note: Check if model expects dB (10*log) or just log. Librosa power_to_db uses 10*log10.
        }

        // Slide buffer: Keep the overlap for the next frame
        // Overlap = win_length - hop_length
        size_t overlap = win_length_ - hop_length_;
        std::vector<int16_t> next_buffer;
        next_buffer.insert(next_buffer.end(), 
                           input_buffer_.end() - overlap, 
                           input_buffer_.end());
        input_buffer_ = next_buffer;

        return mel_energies;
    }
};

int main() {
    AudioFrontEnd afe;
    
    // Example: Feed 20ms of silence/noise
    std::vector<int16_t> dummy_input(320, 100); // 20ms at 16kHz
    
    // Warmup (needs enough data for first window)
    // First call might return empty if buffer < win_length
    std::vector<float> features = afe.process_frame(dummy_input.data(), dummy_input.size());
    
    // Feed again to get valid features once buffer is full
    features = afe.process_frame(dummy_input.data(), dummy_input.size());

    if (!features.empty()) {
        std::cout << "Computed " << features.size() << " Mel features." << std::endl;
        std::cout << "Feature[0]: " << features[0] << std::endl;
    }

    return 0;
}
