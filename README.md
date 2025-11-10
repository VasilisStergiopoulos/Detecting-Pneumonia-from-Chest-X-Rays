# \# Detect Pneumonia from Chest X-Rays

# 

# Chest X-ray image classification for pneumonia detection.

# 

# \## TL;DR

# \- \*\*Task:\*\* Classify chest X-ray images (3-class: \*Normal / Bacterial / Viral\*).

# \- \*\*Approach:\*\* Transfer learning with modern CNN/ViT (e.g., ResNet101, DenseNet169, DeiT) + strong medical-image augmentations + class-aware sampling.

# \- \*\*Key extras:\*\* Custom loss functions, label smoothing, custom k-fold cross-validation.

# 

# ---

# 

# \## 1. Competition Overview

# \- \*\*Name:\*\* Detect Pneumonia (Spring 2024)

# \- \*\*Host:\*\* Kaggle (private classroom competition)

# \- \*\*Goal:\*\* Predict pneumonia subtype from chest X-rays (Normal, Bacterial, Viral).

# \- \*\*Metric:\*\* Accuracy

# \- \*\*Submission format:\*\* CSV schema (`image\_id,class\_id`)

# 

# ---

# 

# \## 2. Dataset

# Dataset was provided as part of the course’s Kaggle classroom competition.

# 

# \*\*Training set:\*\* 4,672 chest X-ray images  

# \- Normal: 1,227  

# \- Bacterial Pneumonia: 2,238  

# \- Viral Pneumonia: 1,207  

# 

# \*\*Test set:\*\* 1,168 images (unlabeled)

# 

# \*\*Folder structure\*\*

# Detecting-Pneumonia-From-Chest-X-Rays/

# &nbsp; dataset/

# &nbsp;   train\_images/

# &nbsp;     README.txt                

# &nbsp;   test\_images/

# &nbsp;     README.txt

# &nbsp;   labels\_train.csv             # Total training images (NORMAL/BACTERIAL/VIRAL)

# &nbsp;   train\_labels.csv             # Train split labels

# &nbsp;   val\_labels.csv               # Val split labels





---



\## 3. Method



\### 3.1 Preprocessing \& Data Augmentation

Different transforms are used for training and validation:



\*\*Training transforms\*\*

\- Convert to PIL

\- Resize to 256×256

\- RandomResizedCrop to 224×224

\- RandAugment + AugMix

\- ToTensor, Normalize (ImageNet mean/std)



These augmentations improve generalization by introducing variation in scale, contrast, cropping, brightness and structure.



\*\*Validation transforms\*\*

\- Convert to PIL

\- Resize to 256

\- CenterCrop to 224

\- ToTensor, Normalize



Validation stays deterministic for fair comparison.



---



\### 3.2 Architectures

The main model used in this project was \*\*DeiT-Base (patch16, 224×224)\*\*, a Vision Transformer pretrained on ImageNet.



\- Backbone frozen (for stability / faster convergence)

\- Fine-tuned MLP + classification head

\- CrossEntropyLoss + label smoothing



\*\*Baselines also evaluated\*\*

\- DenseNet-169  

\- EfficientNet-B4  

\- ResNet-101  

\- MobileNet-V3  

\- ShuffleNet-V2  

\- Ensemble (averaged logits)



Each uses a 3-class classifier with softmax output.



---



\### 3.3 Loss \& Optimization

\- Loss: CrossEntropy (optionally ReducedCrossEntropy)

\- Optimizer: AdamW

\- LR schedule: CosineAnnealingLR

\- Batch size: 64

\- Epochs: 200



---



\### 3.4 Cross-Validation

\- k-Fold CV on the train set for robust estimation

\- Stratified to preserve class balance



---



\## 4. Results



This was a 3-class classification problem (Normal / Bacterial / Viral).



The \*\*best performance\*\* was achieved by \*\*DeiT (Vision Transformer)\*\*:



| Model | Accuracy |

|-------|----------|

| \*\*DeiT-Base (Patch16/224)\*\* | \*\*87%\*\* |

| DenseNet-169 | 84% |

| EfficientNet-B4 | 83% |

| ResNet-101 | 82% |

| MobileNet-V3 | 80% |

| ShuffleNet-V2 | 79% |

| Ensemble | 85% |



\*\*Training setup\*\*

\- Frozen DeiT backbone

\- Fine-tuned head

\- AdamW + Cosine scheduler



---



\## 5. Project Structure



Detecting-Pneumonia-from-Chest-X-Rays/

│

├── configs/                     # YAML configuration files for training

│   └── PNEUMONIA.yaml

│

├── dataset/                     # Chest X-ray dataset (train/test)

│   ├── train\_images/

&nbsp;       └── README.txt

│   ├── test\_images/

&nbsp;       └── README.txt

│   ├── labels\_train.csv

│   ├── train\_labels.csv

│   ├── val\_labels.csv

│

├── models/                      # Model architectures

│   ├── deit.py                  

│   ├── densenet.py

│   ├── efficientnet.py

│   ├── mobilenet.py

│   ├── resnet.py

│   ├── shufflenet.py

│   └── ensemble.py

│

├── scripts/                     # Utility scripts 

|   ├── datasets.py              # Dataset loader + transforms

│   ├── inference.py             # Run model on test set 

|   ├── losses.py                # Bregman-based loss functions

│   ├── train.py                 # Main training script

│   └── visualizations.py        # Accuracy/error plots from JSON logs

│

├── README.md

└── LICENSE





---



\## 6. Experiments

\- Image sizes: 224 vs 384 vs 512  

\- Backbone comparison: EffNetV2-S vs DeiT-B/16  

\- Loss comparison: CE vs ReducedCrossEntropy



---



\## 7. Troubleshooting

\- \*\*Overfitting:\*\* stronger augmentation, reduce LR, add weight decay  

\- \*\*Unstable CV:\*\* increase folds, fix seeds, ensure stratified splits  

\- \*\*Low recall on Normal:\*\* class weights, inspect samples, add contrast/brightness aug



---



\## 8. License \& Acknowledgments

\- Code: MIT License  

\- Data: Provided for academic use via classroom competition  

\- Acknowledgments: Kaggle \& course instructors



---



\## 9. Changelog

\- \*\*v1.0.0\*\* — Initial release

