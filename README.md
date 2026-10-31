
# JEPA Ablation Study & Reproduction

This repository contains my independent implementation and ablation study of the Joint Embedding Predictive Architecture (JEPA), including both 2D image (MNIST) and 3D video (MovingMNIST) versions.

## Repository Structure
- `train_2d_jepa.py`: PyTorch implementation of 2D JEPA on MNIST.
- `train_3d_jepa.py`: PyTorch implementation of 3D V-JEPA (spatiotemporal).
- `JEPA_Ablation_Results.csv`: Raw data of the 27-run grid search (3 Architectures × 3 Learning Rates × 3 VICReg Weights).

## Key Implementations
- **Asymmetric Architecture**: EMA (Exponential Moving Average) Target Encoder with stop-gradient.
- **Anti-Collapse**: VICReg (Variance-Covariance Regularization) loss.
- **Evaluation**: Linear Probing on frozen encoder features.

## Note
In the provided code, the dataset saving paths are hardcoded to `C:\Python学习\JEPA`. Please modify the `SAVE_DIR` variable to your local directory before running.
