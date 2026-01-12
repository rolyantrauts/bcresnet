import os
import glob
import wave
import contextlib
import argparse
import random
import torch
import torch.nn as nn
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans, AffinityPropagation
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# --- PANNs Cnn14 Implementation (Miniaturized for single-file usage) ---
# We define the model structure here so you don't need external dependencies
class Cnn14_Embedder(nn.Module):
    def __init__(self, pretrained=True, device='cpu'):
        super(Cnn14_Embedder, self).__init__()
        # Load pre-trained weights from PANNs authors (Zenodo)
        self.device = device
        
        # Standard PANNs CNN14 Architecture
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = self.conv_block(1, 64)
        self.conv_block2 = self.conv_block(64, 128)
        self.conv_block3 = self.conv_block(128, 256)
        self.conv_block4 = self.conv_block(256, 512)
        self.conv_block5 = self.conv_block(512, 1024)
        self.conv_block6 = self.conv_block(1024, 2048)
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        
        if pretrained:
            self.load_weights()

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        )

    def load_weights(self):
        url = 'https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1'
        # Cache path
        home = os.path.expanduser("~")
        cache_dir = os.path.join(home, ".cache/torch/checkpoints")
        os.makedirs(cache_dir, exist_ok=True)
        filename = "Cnn14_mAP=0.431.pth"
        path = os.path.join(cache_dir, filename)
        
        if not os.path.exists(path):
            print(f"Downloading PANNs Cnn14 weights to {path}...")
            torch.hub.download_url_to_file(url, path)
            
        state_dict = torch.load(path, map_location='cpu')
        # We only need the feature extraction layers, not the final classification
        if 'model' in state_dict: state_dict = state_dict['model']
        self.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        # x: (batch, channels, time, freq)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.conv_block6(x)

        x = torch.mean(x, dim=3) # Pool freq
        (x1, _) = torch.max(x, dim=2) # Max pool time
        x2 = torch.mean(x, dim=2)     # Mean pool time
        x = x1 + x2 # Combine
        
        x = self.fc1(x) # 2048-dim embedding
        return x

# --- Main Script ---

def get_args():
    parser = argparse.ArgumentParser(description="Balance audio dataset using PANNs (CNN) Embeddings.")
    parser.add_argument("--root_dir", type=str, required=True, help="Path to the directory containing .wav files")
    # Default set back to 10.0 seconds
    parser.add_argument("--min_duration", type=float, default=10.0, help="Files shorter than this are deleted immediately")
    parser.add_argument("--manual_k", type=int, default=0, help="Set a fixed K (e.g., 10). If 0 (default), uses Fast Auto-Estimation.")
    parser.add_argument("--output_file", type=str, default="delete_list.txt", help="Path to save the list")
    parser.add_argument("--plot_file", type=str, default="balance_report.png", help="Path to save the visualization image")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu or cuda)")
    return parser.parse_args()

def get_wav_duration(file_path):
    try:
        with contextlib.closing(wave.open(file_path, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            return frames / float(rate)
    except:
        return -1

def preprocess_audio(file_path, device='cpu'):
    """Prepares audio for PANNs (Resample to 32k, Mel Spectrogram)"""
    try:
        # PANNs requires 32kHz sample rate
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Resample if needed
        if sample_rate != 32000:
            resampler = torchaudio.transforms.Resample(sample_rate, 32000)
            waveform = resampler(waveform)
            
        # Mix to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Limit duration (take middle chunk if too long, or loop if short)
        target_len = 32000 * 10 # 10 seconds context
        curr_len = waveform.shape[1]
        
        if curr_len > target_len:
            # Take random crop or center
            start = (curr_len - target_len) // 2
            waveform = waveform[:, start:start+target_len]
        elif curr_len < target_len:
            # Pad
            padding = target_len - curr_len
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        waveform = waveform.to(device)
        
        # Compute Log-Mel Spectrogram for PANNs
        # Window size 1024, hop 320, mels 64 (standard PANNs config)
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=32000,
            n_fft=1024,
            win_length=1024,
            hop_length=320,
            n_mels=64,
            f_min=50,
            f_max=14000,
            power=2.0
        ).to(device)(waveform)
        
        mel_spec = torch.log(mel_spec + 1e-7)
        # Shape: (1, 64, Time) -> Transpose to (1, 1, Time, 64) for CNN
        mel_spec = mel_spec.transpose(1, 2).unsqueeze(0)
        
        return mel_spec

    except Exception:
        return None

def generate_balance_plot(original_counts, target_count, output_path):
    clusters = sorted(original_counts.keys())
    orig_values = [original_counts[k] for k in clusters]
    kept_values = [min(v, target_count) for v in orig_values]
    
    x = np.arange(len(clusters))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x, orig_values, width, label='Original Count', color='lightgray')
    plt.bar(x, kept_values, width, label='Retained Count', color='mediumseagreen')
    plt.axhline(y=target_count, color='r', linestyle='--', label=f'Target Cap ({target_count})')
    
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Files')
    plt.title('Dataset Balance Analysis (PANNs CNN Embeddings)')
    plt.xticks(x, clusters)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"\n[Report] Balance chart saved to: {os.path.abspath(output_path)}")

def estimate_best_k_fast(X_scaled, seed):
    n_samples = X_scaled.shape[0]
    subset_size = 1000
    if n_samples > subset_size:
        rng = np.random.RandomState(seed)
        indices = rng.choice(n_samples, subset_size, replace=False)
        X_subset = X_scaled[indices]
    else:
        X_subset = X_scaled

    print(f"\n[Auto-K] Running Affinity Propagation on {len(X_subset)} samples...")
    af = AffinityPropagation(preference=None, random_state=seed, damping=0.9).fit(X_subset)
    k_found = len(af.cluster_centers_indices_)
    
    if k_found > 50: return 50
    if k_found < 2: return 3
    return k_found

def main():
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    device_name = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    device = torch.device(device_name)
    print(f"Using Device: {device_name}")
    
    # Initialize PANNs Model
    print("Loading PANNs Cnn14 Model...")
    embedder = Cnn14_Embedder(pretrained=True, device=device).to(device)
    embedder.eval()
    
    root_dir = args.root_dir
    wav_files = glob.glob(os.path.join(root_dir, "**", "*.wav"), recursive=True)
    total_files_count = len(wav_files)
    
    files_to_delete = []
    valid_files = []      
    valid_features = []   
    count_short = 0

    print(f"Scanning {total_files_count} files in '{root_dir}'...")

    # --- PHASE 1: Extract Embeddings ---
    with torch.no_grad():
        for file_path in tqdm(wav_files, desc="Encoding Audio", unit="file"):
            abs_path = os.path.abspath(file_path)
            duration = get_wav_duration(file_path)
            
            # Check Duration
            if duration < args.min_duration:
                files_to_delete.append(abs_path)
                count_short += 1
                continue
            
            # Extract
            spec = preprocess_audio(file_path, device=device)
            if spec is not None:
                # Forward Pass through CNN
                embedding = embedder(spec) # Shape: (1, 2048)
                feat = embedding.cpu().numpy().squeeze()
                
                if np.isfinite(feat).all():
                    valid_files.append(abs_path)
                    valid_features.append(feat)
                else:
                    files_to_delete.append(abs_path)
                    count_short += 1
            else:
                files_to_delete.append(abs_path)
                count_short += 1 

    # --- PHASE 2: Estimate K & Cluster ---
    num_valid = len(valid_files)
    count_balance_deletions = 0

    if num_valid > 5:
        X = np.array(valid_features)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if args.manual_k > 0:
            final_k = args.manual_k
            print(f"\nUsing Manual K: {final_k}")
        else:
            final_k = estimate_best_k_fast(X_scaled, args.seed)
            print(f"[Auto-K] Suggested optimal clusters: {final_k}")

        print(f"Clustering {num_valid} files into {final_k} groups...")
        kmeans = MiniBatchKMeans(n_clusters=final_k, random_state=args.seed, batch_size=256)
        labels = kmeans.fit_predict(X_scaled)
        
        clusters = {} 
        for i, label in enumerate(labels):
            if label not in clusters: clusters[label] = []
            clusters[label].append(valid_files[i])

        # --- PHASE 3: Balancing ---
        counts = [len(v) for v in clusters.values()]
        val_mean = np.mean(counts)
        val_median = np.median(counts)
        target_count = int(max(val_mean, val_median))
        
        # Stats
        cluster_counts_map = {cid: len(f) for cid, f in clusters.items()}
        final_vals = [min(c, target_count) for c in counts]
        avg_keep_count = np.mean(final_vals)

        # Plot
        generate_balance_plot(cluster_counts_map, target_count, args.plot_file)

        print("\n" + "-"*60)
        print(f"Semantic Analysis Results (PANNs Cnn14)")
        print(f"Mean Cluster Size (Original): {val_mean:.1f}")
        print(f"TARGET COUNT (Cap):           {target_count}")
        print(f"Avg Files Kept per Cluster:     {avg_keep_count:.1f}")
        print("-" * 60)
        print(f"{'Cluster':<8} | {'Count':<8} | {'Action'}")
        print("-" * 60)
        
        for cid in sorted(clusters.keys()):
            count = len(clusters[cid])
            if count > target_count:
                to_remove = count - target_count
                action = f"Trim {to_remove}"
                files = clusters[cid]
                random.shuffle(files)
                extras = files[target_count:]
                files_to_delete.extend(extras)
                count_balance_deletions += len(extras)
            else:
                action = "Keep All"
            print(f"{cid:<8} | {count:<8} | {action}")

    else:
        print("Not enough files to perform clustering.")

    # --- PHASE 4: Save ---
    if files_to_delete:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for item in files_to_delete:
                if ' ' in item:
                    f.write(f'"{item}"\n')
                else:
                    f.write(f"{item}\n")
        print(f"\nSaved list to: {os.path.abspath(args.output_file)}")
    else:
        print("\nNo files need deletion.")

    # Stats
    print("\n" + "="*40)
    print("FINAL SUMMARY")
    print("="*40)
    print(f"Files Analysed:     {num_valid} / {total_files_count}")
    print(f"Short/Error Files:  {count_short}")
    print(f"Excess Files:       {count_balance_deletions}")
    print("-" * 40)
    print(f"TOTAL TO DELETE:    {len(files_to_delete)}")
    print("="*40)

if __name__ == "__main__":
    main()
