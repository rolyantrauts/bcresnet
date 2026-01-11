📚 Multi-Stage Dataset & Curriculum Training StrategyThis document outlines the dataset organization and training strategy required to build a production-grade Wakeword model using BCResNet.To achieve high accuracy and low false acceptance rates (FAR), we do not simply dump all data into the model at once. Instead, we organize data into specific "buckets" and use a Multi-Stage Curriculum to bias the training distribution as the model matures.1. Dataset Organization: The "Bucket" StructureOrganize your raw audio files into three primary root classes. Crucially, the Unknown class is subdivided to allow for specific biasing of adversarial examples.Plaintextdatasets/
├── 01_positive/              # The Target Wakeword
│   ├── clean_near_field/     # High quality studio/close-mic recordings
│   └── far_field/            # Real-world recordings (distance > 2m)
│
├── 02_negative/              # "Unknown" Speech (NOT the wakeword)
│   ├── general_speech/       # Standard datasets (LibriSpeech, Common Voice)
│   └── adversarial/          # "Hard Negatives" (Phonetically similar words)
│       ├── hey_computer/     # if target is "Hey Computer" -> "Hey Commuter"
│       ├── okay_google/      # if target is "Okay Google" -> "Okay Poodle"
│       └── phonemes/         # Isolated sounds similar to target start/end
│
├── 03_background/            # Non-speech Audio
│   ├── silence/              # Pure digital silence or extremely quiet room
│   ├── domestic_noise/       # Vacuum, typing, door slams, footsteps
│   └── music/                # Instrumental tracks (highly dynamic)
│
└── augmentation/             # Generated Acoustic Environments
    ├── rirs_voice/           # [See RIR Generator.md]
    └── rirs_noise/           # [See RIR Generator.md]
🧠 The Adversarial BucketAdversarial data is critical for reducing False Positives. If your wakeword is "Hey Jarvis", your adversarial bucket should contain:Rhymes: "Hey Harvest", "Pay Service".Partial Overlaps: "Jar", "Vis", "Hey".Confusing Contexts: Sentences containing these words naturally.2. Dynamic Augmentation PipelineWe utilize the Domestic RIR Generator to apply acoustic characteristics on the fly.The Pipeline:Select Raw Audio: Pick a file from Positive, Negative, or Adversarial buckets.Apply RIR: Convolve with a domestic_voice RIR (for speech) or domestic_noise RIR (for background).Mix Noise: Add 03_background noise at a specific SNR (Signal-to-Noise Ratio).Normalize: Scale audio to target loudness.3. Curriculum Training: Biasing Data Over TimeTraining is split into 3 Stages. We bias the data loader probabilities and augmentation severity to guide the model from learning "what the word sounds like" to "how to distinguish it in 5dB SNR with reverb."🟢 Stage 1: Warmup (Epochs 0 - 10)Goal: Rapid convergence. Teach the model the basic phonetic structure of the wakeword.Data Bias: High percentage of Clean Positives.Adversarial: Disabled or very low probability.Augmentation: Minimal (High SNR, no Reverb).ParameterValuePositive Ratio40%General Negative50%Adversarial10%Silence/Noise0%RIR Probability0% (Clean only)SNR Range20dB - 30dB (Very Clear)Learning Rate0.005🟡 Stage 2: Robustness (Epochs 11 - 50)Goal: Generalization. Introduce the "Domestic" environment using your RIR dataset.Data Bias: Balanced distribution.Augmentation: Full Domestic RIRs enabled. Moderate Noise.ParameterValuePositive Ratio30%General Negative40%Adversarial20%Silence/Noise10%RIR Probability80% (Apply Domestic RIRs)SNR Range5dB - 15dB (Typical Home)Learning Rate0.005 (Reduce to 0.001 if loss spikes)🔴 Stage 3: Hard Mining (Epochs 51+)Goal: False Positive Rejection. Force the model to distinguish "Hey Jarvis" from "Hey Harvest" in noisy rooms.Data Bias: Heavily biased towards Adversarial negatives.Augmentation: Aggressive. Low SNR.ParameterValuePositive Ratio25%General Negative25%Adversarial40% (Focus on hard cases)Silence/Noise10%RIR Probability100%SNR Range-5dB - 10dB (Noisy Party/TV)Learning Rate0.001 (Fine-tuning)4. Implementation GuideTo implement this in bcresnet/main3.py, you can modify the WeightedRandomSampler weights or the dataset configuration between epochs.Conceptual Implementation (Python)Python# In your DataLoader / Training Loop
def get_stage_config(epoch):
    if epoch < 10:
        return {
            "p_positive": 0.4, "p_adversarial": 0.1, 
            "snr_min": 20, "rir_prob": 0.0
        }
    elif epoch < 50:
        return {
            "p_positive": 0.3, "p_adversarial": 0.2, 
            "snr_min": 5, "rir_prob": 0.8
        }
    else:
        return {
            "p_positive": 0.25, "p_adversarial": 0.4, 
            "snr_min": -5, "rir_prob": 1.0
        }

# Inside training loop
for epoch in range(MAX_EPOCHS):
    cfg = get_stage_config(epoch)
    
    # Update Dataset parameters dynamically
    train_dataset.set_class_probabilities(
        pos=cfg['p_positive'], 
        adv=cfg['p_adversarial']
    )
    train_dataset.set_augmentation_params(
        snr_db=cfg['snr_min'], 
        rir_prob=cfg['rir_prob']
    )
    
    # Train...
Tips for bcresnet Repo UsersCheckpointing: When transitioning from Stage 1 to Stage 2, ensure you save a checkpoint. If Stage 2 training diverges (Loss = NaN) due to the sudden difficulty spike, reload the Stage 1 checkpoint and lower the Learning Rate to 0.001.Exporting: Always perform your final export (using --export-only) using the weights from Stage 3, as these will be most robust to real-world triggers.
