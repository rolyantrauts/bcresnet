🏠 Domestic Room Impulse Response (RIR) GeneratorThis utility generates a dataset of 10,000 Room Impulse Responses (RIRs) specifically designed to mimic domestic environments and smart speaker usage scenarios.  
Unlike generic RIR generators that often place microphones in the center of empty rooms, this script focuses on realistic constraints:Rooms: Living rooms, bedrooms, kitchens ($3m \times 3m$ to $8m \times 8m$).  
Acoustics: Furnished rooms with realistic $T_{60}$ times ($0.2s - 0.6s$).Placement Logic: Microphones are placed on shelves or tables near walls, rather than floating in mid-air.  
🎯 Key FeaturesRealistic Domestic Rooms: Simulates dimensions typical of residential spaces.  
Smart Speaker Placement: Forces microphone placement near walls ($0.1m - 0.5m$) at table/shelf height ($0.7m - 1.2m$), significantly affecting early reflections.  
Dual RIR Generation: Generates separate RIR files for Voice and Noise for the same room configuration.  
Why? This allows dynamic mixing during training. You can mix the Voice at 0dB and Noise at -5dB accurately because they share the exact same acoustic room properties.  
Source Constraints:Voice: Placed at sitting/standing height ($1.0m - 1.8m$) with a minimum distance check to avoid "inside the mic" artifacts.Noise: Placed randomly (e.g., simulating TV, window, appliances).🛠 PrerequisitesYou will need the following Python libraries:
```
pip install numpy pyroomacoustics scipy tqdm audiomentations
```
📄 Generation ScriptSave the following code as generate_rirs.py and run it to create your dataset.  
Python
```
import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
import os
from tqdm import tqdm
import random

# ==========================================
# CONFIGURATION
# ==========================================
NUM_RIRS = 10000
FS = 16000  # Sample rate (standard for speech)
OUTPUT_DIR = "domestic_rirs_dataset"

# Room Dimensions (Meters) - Domestic ranges
# Small bedroom (3x3) to Large Living Room (8x8)
ROOM_DIM_RANGES = {
    'length': (3.0, 8.0),
    'width':  (3.0, 8.0),
    'height': (2.4, 3.0)  # Standard ceiling height
}

# Reverberation Time (T60) in seconds
# 0.2 (Bedroom with carpet/curtains) to 0.6 (Living room with tiles/wood)
RT60_RANGE = (0.2, 0.6)

# Smart Speaker constraints
MIC_HEIGHT_RANGE = (0.7, 1.2)  # Table/Shelf height
MIC_WALL_DIST = (0.1, 0.5)     # Distance from wall (10cm to 50cm)

# Source constraints
VOICE_HEIGHT_RANGE = (1.0, 1.8) # Sitting to Standing
MIN_VOICE_DIST = 1.0            # Voice shouldn't be inside the mic

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_random_room_dims():
    l = np.random.uniform(*ROOM_DIM_RANGES['length'])
    w = np.random.uniform(*ROOM_DIM_RANGES['width'])
    h = np.random.uniform(*ROOM_DIM_RANGES['height'])
    return np.array([l, w, h])

def get_smart_speaker_pos(room_dim):
    """
    Places microphone near a wall (mimicking a shelf, sideboard, or counter).
    """
    l, w, h = room_dim
    
    # Pick a random wall: 0=North, 1=South, 2=East, 3=West
    wall = random.randint(0, 3)
    dist = np.random.uniform(*MIC_WALL_DIST)
    z = np.random.uniform(*MIC_HEIGHT_RANGE)
    
    if wall == 0:   # Near x=0
        pos = [dist, np.random.uniform(dist, w-dist), z]
    elif wall == 1: # Near x=L
        pos = [l-dist, np.random.uniform(dist, w-dist), z]
    elif wall == 2: # Near y=0
        pos = [np.random.uniform(dist, l-dist), dist, z]
    else:           # Near y=W
        pos = [np.random.uniform(dist, l-dist), w-dist, z]
        
    return np.array(pos)

def get_source_pos(room_dim, mic_pos, is_voice=True):
    """
    Places a source ensuring it's within the room and (if voice) 
    not too close to the mic.
    """
    l, w, h = room_dim
    max_attempts = 100
    
    for _ in range(max_attempts):
        # Random X, Y
        x = np.random.uniform(0.5, l-0.5) # Keep away from walls slightly
        y = np.random.uniform(0.5, w-0.5)
        
        # Z height depends on source type
        if is_voice:
            z = np.random.uniform(*VOICE_HEIGHT_RANGE)
        else:
            z = np.random.uniform(0.5, h-0.5) # Noise can be anywhere (TV, Window)
            
        pos = np.array([x, y, z])
        
        # Check distance constraint for voice
        if is_voice:
            dist = np.linalg.norm(pos - mic_pos)
            if dist < MIN_VOICE_DIST:
                continue
                
        return pos
    
    return pos # Fallback

print(f"Generating {NUM_RIRS} RIRs to {OUTPUT_DIR}...")

for i in tqdm(range(NUM_RIRS)):
    # 1. Generate Room Geometry
    room_dim = get_random_room_dims()
    rt60 = np.random.uniform(*RT60_RANGE)
    
    # Estimate absorption required for this RT60
    # We use Sabin's formula approximation via pyroomacoustics inverse
    try:
        e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
        
        # Create Room
        room = pra.ShoeBox(
            room_dim, 
            fs=FS, 
            materials=pra.Material(e_absorption), 
            max_order=max_order
        )
        
        # 2. Place Smart Speaker (Microphone)
        mic_pos = get_smart_speaker_pos(room_dim)
        room.add_microphone(mic_pos)
        
        # 3. Place Voice Source (Source 0)
        voice_pos = get_source_pos(room_dim, mic_pos, is_voice=True)
        room.add_source(voice_pos)
        
        # 4. Place Noise Source (Source 1)
        # Optional: Add multiple noise sources if needed
        noise_pos = get_source_pos(room_dim, mic_pos, is_voice=False)
        room.add_source(noise_pos)
        
        # 5. Compute RIR
        room.compute_rir()
        
        # 6. Extract and Save RIRs
        # room.rir is a list of lists: [mic_channel][source_index]
        
        # Save Voice RIR
        voice_rir = room.rir[0][0]
        # Normalize to prevent clipping
        voice_rir = voice_rir / np.max(np.abs(voice_rir)) 
        filename_voice = os.path.join(OUTPUT_DIR, f"rir_{i:05d}_voice_rt{rt60:.2f}.wav")
        wavfile.write(filename_voice, FS, voice_rir.astype(np.float32))
        
        # Save Noise RIR
        noise_rir = room.rir[0][1]
        noise_rir = noise_rir / np.max(np.abs(noise_rir))
        filename_noise = os.path.join(OUTPUT_DIR, f"rir_{i:05d}_noise_rt{rt60:.2f}.wav")
        wavfile.write(filename_noise, FS, noise_rir.astype(np.float32))

    except Exception as e:
        print(f"Error generating RIR {i}: {e}")
        continue

print("Done!")
```
🚀 Usage with AudiomentationsOnce you have generated the folder of RIRs (e.g., domestic_rirs_dataset), you can integrate them into your training pipeline using audiomentations.Pythonfrom audiomentations import ApplyImpulseResponse, Compose
```
augment = Compose([
    ApplyImpulseResponse(
        ir_path="domestic_rirs_dataset", # Path to the folder created above
        p=0.5
    ),
    # You can combine this with background noise injection
    # AddBackgroundNoise(...)
])

# Example Usage
# augmented_audio = augment(samples=audio_waveform, sample_rate=16000)
```
