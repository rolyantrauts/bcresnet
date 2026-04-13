```
((venv) ) stuartnaylor@Stuarts-Mac-mini bcresnet % python main.py --data_root ../gkws/data2 --n_mels 80 --clip_duration=1.4 --lr=0.002 --epochs=100 --patience=7 --batch_size=512 --tau=2 --arcface

========================================
   TRAINING CONFIGURATION (External)   
========================================
Device       : auto
Clip Duration: 1.4s (22400 samples)
Mel Bins     : 80
Spec Shape   : [1, 1, 80, 141] (Input for C++)
Model Tau    : 2.0 (SSN=True)
Dropout      : 0.3
ArcFace      : ON (Outputting Embeddings)
SpecAug Prob : 80.0%
Class Index  : Auto-detect
Start Epoch  : 1 / 100
========================================

Running on device: mps
Loading data from ../gkws/data2...
Found 502369 files for training in 3 classes.
Found 37815 files for validation in 3 classes.
Found 37815 files for testing in 3 classes.
Detected Classes: ['hey_jarvis', 'noise', 'unknown'] (Total: 3)
📄 Class mapping saved to index.txt
Building BCResNet-2.0...
Total Parameters: 112,156
```
