# Glioma-IDH-Deep-Radiomics
This repository implements a deep learning-based framework for the automatic classification of glioma IDH mutation status using multimodal MRI scans and radiomic feature extraction.


## Overview
The proposed pipeline integrates deep learning and radiomics to provide accurate and interpretable prediction of glioma IDH status. The method is designed to support clinical decision-making, prognosis estimation, and early risk stratification.


## Pipeline
1. MRI preprocessing (normalization, registration)
2. Automatic tumor segmentation using state-of-the-art deep learning models
3. Radiomic feature extraction (intensity, texture, shape)
4. CNN model training
5. Model evaluation

![Pipeline overview](images/pipeline.png)