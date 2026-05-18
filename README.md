# Breast Ultrasound Segmentation and Classification

This repository contains the cleaned code pack for the Computer Science Studio 2 portfolio project. It combines the earlier breast cancer segmentation work with the later Studio 2 classification experiments.

The project was developed mainly in **Google Colab**. The shared Colab notebooks are the main code artifacts because most training, testing, debugging, and peer review happened through Colab notebooks and Google Drive.

## Start Here

The most important code file is:

```text
01_breast_cancer_segmentation/integrated_unet_efficientnet_v2.ipynb
```

This is the final integrated notebook. It should be reviewed first because it contains the final project pipeline:

1. **Stage A: Attention U-Net segmentation** for breast ultrasound lesion masks.
2. **Stage B: `.npz` bridge** for saving images, predicted masks, ground-truth masks, and labels.
3. **Stage C: EfficientNet-B0 classification** with soft-guided spatial attention.
4. Final comparison between predicted-mask, GT-mask oracle, no-mask attention, and pure EfficientNet settings.

For the full assessment, the final written report is the main written submission. For this GitHub repository, the main code evidence is `integrated_unet_efficientnet_v2.ipynb`.

## Repository Structure

```text
.
├── 01_breast_cancer_segmentation/
│   ├── breast-cancer-image-segmentation-attention-unet.ipynb
│   ├── breast-cancer-segmentation-attention-unet-colab.ipynb
│   ├── integrated_unet_efficientnet_v1.ipynb
│   ├── integrated_unet_efficientnet_v2.ipynb
│   └── integrated_v1/
│       ├── classifier.pth
│       ├── classifier_v2.pth
│       ├── classifier_v2_nomask.pth
│       ├── classifier_v2_pure.pth
│       └── split_manifest.json
├── 02_computer_science_studio2/
│   ├── README_local_run.md
│   ├── best_densenet121_v2.pth
│   ├── integrated_notebook.ipynb
│   ├── integrated_notebook.py
│   ├── integrated_notebookv2.ipynb
│   ├── integrated_notebookv2.py
│   ├── integrated_notebookv2-fix.py
│   ├── integrated_notebookv2.original-colab-backup.ipynb
│   ├── integrated_notebookv3.py
│   ├── monai_DenseNet121_2.ipynb
│   ├── monai_DenseNet121_2_local.py
│   ├── monai_DenseNet121ipynb.ipynb
│   ├── monai_densenet121ipynb.py
│   ├── preprocessing_pipeline_v2.png
│   └── requirements.txt
└── README.md
```

## File Guide

`01_breast_cancer_segmentation/` is the earlier and most important project folder. It contains the final segmentation-guided pipeline.

`integrated_unet_efficientnet_v2.ipynb` is the final notebook and should be used as the main reference for marking or code review.

`integrated_unet_efficientnet_v1.ipynb` is an earlier integrated attempt. It is kept to show development progress and debugging, but it is not the final version.

`breast-cancer-segmentation-attention-unet-colab.ipynb` and `breast-cancer-image-segmentation-attention-unet.ipynb` are segmentation-focused notebooks from the earlier stage of the project.

`02_computer_science_studio2/` contains the later Studio 2 work, including MONAI DenseNet121 experiments, notebook-to-script exports, local run notes, and supporting material. These files are useful background, but they are not the main final pipeline.



## Project Summary

The final project investigates breast ultrasound analysis using a two-stage deep learning workflow. First, Attention U-Net predicts a soft lesion mask. Then EfficientNet-B0 uses the predicted mask as a soft attention guide for benign/malignant classification.

The final result is not claimed as a clinical system. The main value of the project is showing a complete Colab-based workflow: segmentation, mask handoff, classification, ablation comparison, and visual explanation.
