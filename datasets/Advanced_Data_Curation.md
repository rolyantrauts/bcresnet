🔬 Advanced Data Curation: Spectral Framing & Phonetic Balancing  
This document details advanced strategies for formatting input audio to maximize feature extraction in the BCResNet model.  
It covers how to utilize "spectral whitespace" (padding), optimal positioning for adversarial data, and generating phonetically balanced datasets using Text-to-Speech (TTS).  
The Strategy: Spectral Imaging & WhitespaceA Convolutional Neural Network (CNN) like BCResNet does not "hear" sound; it "sees" an image (the Spectrogram).  
How you frame the object within that image drastically changes how the model learns.  
The "Whitespace" Advantage for Wakewords  
When training a Wakeword (Positive Class), it is often advantageous to include surrounding "whitespace" (silence or ambient room noise) as part of the learned signature.  
The Concept: A wakeword is rarely a continuous stream of signal.  
It has an Onset (Silence $\to$ Speech) and an Offset (Speech $\to$ Silence).  
Implementation: instead of cropping the audio tightly to the word boundary (len(word)), crop to len(word) + 200ms_padding.Why? This forces the model to learn the transition features.  
It learns that "Hey Computer" is defined not just by the phonemes, but by the distinctive burst of energy rising from the noise floor.  
Centering Adversarial DataConversely, for Adversarial/Negative data, you want the model to ignore edge effects and focus purely on the internal phonetic signature.  
The Problem: If all your "Negative" clips start exactly when the word starts (t=0), the model might overfit to the "beginning of the file" rather than the word itself.  
The Fix: Embed adversarial words centrally within a short phrase or surround them with significant padding.  
Effect: The model scans the spectrogram and sees similar patterns (phonemes) but learns that without the specific Onset/Offset signature of the Wakeword, it must remain inactive.2.  
Phonetic Balancing with TTSTo create a robust "Hard Negative" bucket, we cannot rely solely on real datasets. We use Text-to-Speech (TTS) to generate words that are phonetically adjacent to our target but distinct enough to be negatives.  
Grouping by Consonant ClassSplit your adversarial generation into phonetic groups based on the Consonant Manner of Articulation.  
Example Target: "Hey Jarvis" ( /h/ /eɪ/ /ʤ/ /ɑ/ /r/ /v/ /ɪ/ /s/ )  
Group A: Plosive/Affricate Swaps (The "J" Attack)Target Sound: /ʤ/ (Start of "Jarvis")  
Adversarial Swaps: Replace with /ʧ/ (Ch), /d/ (D), /g/ (G).TTS Words: "Harvest", "Garbage", "Chargers".  
Group B: Fricative Swaps (The "V/S" Attack)Target Sound: /v/ /s/ (End of "Jarvis")Adversarial Swaps: Replace with /f/ (F), /z/ (Z), /θ/ (Th).  
TTS Words: "Service", "Surface", "Nervous".Group C: Vowel ShiftsTarget Sound: /ɑ/ (Ah)Adversarial Swaps: Replace with /æ/ (Cat), /oʊ/ (Go).  
TTS Words: "Joyous", "Jovis".Generating "Padding" WordsUse TTS to generate neutral "filler" words to act as padding for the "Centering" strategy mentioned in Section 1.Fillers: "The", "And", "It", "For".  
Construction: [Filler] + [Adversarial] + [Filler] $\to$ "The Harvest is..."3. Revised Training Curriculum (The "Warm-Up")Integrating these concepts into the curriculum ensures the model establishes a solid baseline before tackling complex spectral images.  
🟢 Stage 1: The "Clean" Warm-Up (Epoch 0-5)
Goal: Lock onto the primary spectral signature.  
Positives: Clean Wakewords with spectral whitespace (200ms padding).  
Negatives: Non-adversarial Unknowns (General speech, distinct words).  
Adversarial: 0%. Do not confuse the model yet.Augmentation: None.  
🟡 Stage 2: Structural Introduction (Epoch 6-20)  
Goal: Differentiate structure.  
Positives: Wakewords with variable padding (randomized shift).  
Negatives: Introduce TTS Adversarials (Group A & B), but placed centrally in silence.  
Augmentation: Light background noise (20dB SNR).  
🔴 Stage 3: The "Real World" (Epoch 21+)  
Goal: Boundary refinement.  
Positives: Domestic RIR applied. Heavy noise overlap.  
Negatives: Adversarials embedded in sentences ("The Harvest is ready").  
Augmentation: Full 20ms Streaming Simulation (random slicing of the input window).  
Python Implementation: TTS GeneratorUse this script to generate your phonetically balanced adversarial dataset.
```
Pythonimport os
import pyttsx3 # pip install pyttsx3

# Configuration
OUTPUT_DIR = "./datasets/02_negative/adversarial_tts"
TARGET_WAKEWORD = "Jarvis"

# Phonetic Groups (Modify for your Wakeword)
ADVERSARIAL_GROUPS = {
    "plosive_confusions": ["Harvest", "Garbage", "Chargers", "Carver"],
    "fricative_confusions": ["Service", "Surface", "Nervous", "Elvis"],
    "rhymes": ["Starfish", "Parsley", "Artist"]
}

PADDING_WORDS = ["The", "And", "It", "Is", "A"]

def generate_tts_dataset():
    engine = pyttsx3.init()
    
    # Configure Voice (Try to get multiple voices if OS permits)
    voices = engine.getProperty('voices')
    
    for group_name, words in ADVERSARIAL_GROUPS.items():
        group_path = os.path.join(OUTPUT_DIR, group_name)
        os.makedirs(group_path, exist_ok=True)
        
        for word in words:
            for i, voice in enumerate(voices):
                engine.setProperty('voice', voice.id)
                
                # 1. Generate Isolated (Centered in Silence)
                # We add silence by saving just the word, manual padding happens in DataLoader
                filename = os.path.join(group_path, f"{word}_isolated_v{i}.wav")
                engine.save_to_file(word, filename)
                
                # 2. Generate Embedded (Sentence Context)
                # "The [Word] is"
                phrase = f"The {word} is"
                filename_ctx = os.path.join(group_path, f"{word}_context_v{i}.wav")
                engine.save_to_file(phrase, filename_ctx)
                
        engine.runAndWait()

if __name__ == "__main__":
    generate_tts_dataset()
    print("Adversarial TTS Dataset Generated.")
```
