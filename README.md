# AI-Generated Image Detection

Course project for Advanced Machine Learning at Dublin City University, co-authored with Thuso Sehloho. It is a binary image classification task: given a product-listing photo, predict whether it is **AI-generated (1)** or **authentic (0)**. The metric is **F1**.

The dataset is not included in this repository.

## Approach

A hybrid CNN-SVM. EfficientNetV2-S (ImageNet-21k pretrained) is fine-tuned with an augmentation pipeline that simulates image degradation (JPEG compression and resizing). The fine-tuned backbone is then used as a feature extractor for an RBF-kernel SVM. Evaluation uses 5-fold cross-validation, and test predictions average the 5 fold models.

**Headline result:** F1 = **0.908 ± 0.014** (recall 0.918, AUC-ROC 0.967) in 5-fold cross-validation.

## Results

Progression of validation F1:

| Method | F1 |
|---|---|
| Baseline (Random Forest) | 0.636 |
| ResNet50 (fine-tuned) + LightGBM | 0.804 |
| ResNet50 (fine-tuned + augmentation) + LightGBM | 0.830 |
| DINOv2 + ConvNeXt-V2 (frozen) + SVM | 0.841 |
| EfficientNetV2-S (frozen) + Logistic Regression | 0.871 |
| EfficientNetV2-S (frozen) + SVM | 0.877 |
| EfficientNetV2-S (fine-tuned) | 0.881 |
| EfficientNetV2-S (fine-tuned) + SVM | 0.890 |
| EfficientNetV2-S (fine-tuned + augmentation) | 0.897 |
| EfficientNetV2-S (fine-tuned + augmentation) + XGBoost | 0.900 |
| **EfficientNetV2-S (fine-tuned + augmentation) + SVM** | **0.908** |

ImageNet-21k pretraining and a CNN backbone generalise better than a ViT on a small dataset. Fine-tuning the backbone and adding augmentation gave the largest gains.

## Report

The full write-up is available in `Report.pdf`. Dataset details and example images have been omitted.

## Notebooks

```
notebooks/
├── exploration.ipynb                  # data and feature-space exploration
├── resnet50_lightgbm_baseline.ipynb   # ResNet50 + LightGBM baseline
├── dino_convnext_svm.ipynb
├── efficientnetv2_fine_tuning.ipynb
├── efficientnetv2_svm.ipynb           # best pipeline + test inference
└── efficientnetv2_xgb.ipynb
```

## Running

The notebooks were written for Google Colab, with data mounted from Drive. The baseline was run locally. Main dependencies: `torch`, `timm`, `scikit-learn`, `xgboost`, `lightgbm`, `pandas`, `matplotlib`. Paths at the top of each notebook need adjusting to your setup.
