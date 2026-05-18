# Breast Cancer Segmentation and Computer Science Studio 2

This repository merges two project folders into one cleaned GitHub-ready workspace:

1. `01_breast_cancer_segmentation/` - the original breast cancer segmentation work, which came first in the project timeline.
2. `02_computer_science_studio2/` - the later Computer Science Studio 2 work, including classification notebooks, MONAI DenseNet121 experiments, local scripts, and presentation material.

## Important Colab Note

Throughout this project, our main development and collaboration workflow used **Google Colab**. Code was written, tested, shared, and reviewed through Colab notebooks and Colab sharing links. The `.ipynb` files in this repository should therefore be treated as the primary project artifacts.

## Repository Structure

```text
.
├── 01_breast_cancer_segmentation/
│   ├── breast-cancer-segmentation-attention-unet-colab.ipynb
│   ├── breast-cancer-segmentation-attention-unet-final.ipynb
│   ├── integrated_unet_efficientnet_v1.ipynb
│   ├── integrated_unet_efficientnet_v2.ipynb
│   └── integrated_v1/
├── 02_computer_science_studio2/
│   ├── integrated_notebook*.ipynb
│   ├── integrated_notebook*.py
│   ├── monai_DenseNet121*.ipynb
│   ├── monai_DenseNet121*_local.py
│   ├── requirements.txt
│   └── breast-cancer-ppt.tsx
└── README.md
```

## Running the Notebooks

The recommended way to run the project is still Google Colab:

1. Upload or open the required `.ipynb` notebook in Colab.
2. Mount Google Drive if the dataset or trained model weights are stored there.
3. Install any missing dependencies inside the notebook.
4. Update dataset paths to point to the mounted Colab/Drive location.
5. Run cells in order.

For local Studio 2 experiments, install dependencies from:

```bash
pip install -r 02_computer_science_studio2/requirements.txt
```

The local scripts expect dataset folders to be supplied separately because datasets were removed from the GitHub repository.

## Project Summary

The first part of the project focuses on breast cancer ultrasound image segmentation using U-Net style approaches and integrated classifier experiments. The second part extends the work into Computer Science Studio 2 deliverables, including MONAI DenseNet121 classification experiments, notebook-to-script exports, and supporting presentation material.

