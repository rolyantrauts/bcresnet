```
((venv) ) stuartnaylor@Stuarts-Mac-mini bcresnet % python main.py --data_root ../gkws/data2 --n_mels 80 --clip_duration=1.4 --lr=0.002 --epochs=200 --patience=7 --batch_size=256 --tau=4 --arcface

========================================
   TRAINING CONFIGURATION (External)   
========================================
Device          : auto
Clip Duration   : 1.4s (22400 samples)
Mel Bins        : 80
Spec Shape      : [1, 1, 80, 141] (Input for C++)
Model Tau       : 4.0 (SSN=True)
Dropout         : 0.3
Label Smoothing : 0.0
ArcFace         : ON (Outputting Embeddings)
SpecAug Prob    : 80.0%
Class Index     : Auto-detect
Start Epoch     : 1 / 200
========================================

Running on device: mps
Loading data from ../gkws/data2...
Found 502369 files for training in 3 classes.
Found 37815 files for validation in 3 classes.
Found 37815 files for testing in 3 classes.
Detected Classes: ['hey_jarvis', 'noise', 'unknown'] (Total: 3)
```
