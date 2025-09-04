# 1.Title - Reliable Classification of Imbalanced Lung Cancer Data with Enhanced Split Conformal Prediction (ESCP)
This repository implements the ***Enhanced Split Conformal Prediction (ESCP)*** framework for reliable classification of imbalanced lung cancer datasets. The ESCP algorithm combines several advanced techniques, including ***Sure Independence Screening (SIS)***, ***Synthetic Minority Over-sampling Technique (SMOTE)***, and ***Split Conformal Prediction (SCP)***, to address the challenges of high-dimensional and imbalanced medical data. It aims to improve classification accuracy, especially for minority classes, and provide statistically reliable predictions for clinical applications.

## 2.Description
### 2.1 Code Overview
This repository contains a pipeline for preprocessing, feature selection, and predictive modeling on high-dimensional gene expression datasets.For detailed instructions on how to run the code, please refer to the ***code information*** section.

### 2.2 dataset overview
In the experimental section of this study, eight public gene expression datasets were used. The target variables of these datasets include whether the subject has lung cancer, whether it is small cell lung cancer, and so on. These datasets are sourced from the Kaggle platform, TCGA platform, and datasets from literature(See the reference). The URLs corresponding to each dataset are provided in the ***dataset information*** section.

## 3.Dataset Information:
The eight publicly available datasets used in this study can be accessed through the following URLs:
[IM101](https://www.kaggle.com/datasets/noepinefrin/tcga-lusc-lung-cell-squamous-carcinoma-gene-exp)
[IM102](https://www.kaggle.com/datasets/josemauricioneuro/lung-cancer-patients-mrna-microarray)
[IM103](https://www.kaggle.com/datasets/gchan357/mirna-expression-in-human-lung-cancers/data)
[IM104,6,8](https://www.kaggle.com/datasets/shibumohapatra/icmr-data)
[IM105](https://portal.gdc.cancer.gov/analysis_page?app=Projects)
[IM107](https://github.com/liusulizzu/cancer_prediction_tcga)
These eight datasets have different features and scales, making them suitable for evaluating classification performance on high-dimensional imbalanced datasets. Data preprocessing and feature selection were performed using variance filtering and Sure Independence Screening (SIS) methods, removing low-variance features and retaining those highly correlated with the target variable.

The specific characteristics of the datasets are as follows:

***Lung genedata（IM101）:*** Focuses on lung squamous cell carcinoma (LUSC). Contains data from 551 patients, 321
each with 56,907 TPM-normalized gene expressions. Class imbalance: 502 cancer vs. 49 healthy. Used 322
for analyzing LUSC gene features and classification. 323

***complete dataframe（IM102）:*** From five medical centers, includes 442 samples with over 23,000 gene 324
expressions.The target is survival for more than 18 months; high-risk is defined based on this criterion.. 325
High feature dimension, suitable for feature selection to avoid overfitting. 326

***miRNA lung（IM103）:*** miRNA data for small cell (SCLC) and non-small cell lung cancer (NSCLC). 119 327
NSCLC and 49 SCLC cell lines, with 743 miRNA features. 328

***data（IM104）, KIPC LUAD(IM106), PRAD LUAD(IM108):***  From ICMR, includes multiple cancer types (breast, kidney, 329
colon, lung). 802 samples, each with over 20,000 gene expressions. Used for multi-class classification of 330
cancer types. 331

***Count matrix(IM105):*** 60,660 genes across 600 samples—317 normal, 283 lung cancer. 332
***icgc LUAD(IM107):*** Lung adenocarcinoma data; 543 lung cancer and 55 normal samples, 19,565 genes

Each dataset contains high-dimensional gene expression data with imbalanced classes, reflecting the challenges in lung cancer classification tasks.

## 4.Code information
The workflow is modularized into several scripts:

***Target_Variable_8.py – Variance Filtering***

Removes low-variance features from the raw dataset using a threshold-based filter.

Input: original dataset (.csv) with target variable in the first column.

Output: filtered dataset (*_filtered_gene_data.csv).

***SIS_8.py – Feature Selection (Sure Independence Screening / ANOVA F-test)***

Selects the top-K features most associated with the target variable.

Input: variance-filtered dataset.

Output: SIS-selected dataset (*_filtered_gene_data_SIS.csv).

***Modeling and Conformal Prediction***

Several scripts implement predictive modeling and uncertainty quantification using different classifiers:

RF.py-Random Forest

RFS.py-Random Forest with SMOTE balancing and 5-fold cross-validation

RFSCP.py – Random Forest + Split Conformal Prediction (SCP)

SVM.py-Support Vector Machine (SVM

SVMS.py-SVM with SMOTE balancing and 5-fold cross-validation

SVMSCP.py – Support Vector Machine (SVM) + SCP

XGBoost.py-XGBoost

XGBS.py – XGBoost with SMOTE balancing and 5-fold cross-validation

XGBSCP.py – XGBoost + SCP


These scripts train models on the SIS-selected dataset, evaluate performance (Accuracy, F1-score, AUC, etc.), and visualize results (ROC/PR curves, confusion matrices, calibration plots).

Outputs are saved in results/ or model_evaluation_plots/ directories.

## 5.Usage Instuctions
Variance Filtering → *_filtered_gene_data.csv

SIS Feature Selection → *_filtered_gene_data_SIS.csv

Model Training + SCP → evaluation metrics and visualizations
## 6.Requirements-Any dependencies
contourpy==1.3.2

cycler==0.12.1

fonttools==4.58.4

imbalanced-learn==0.13.0

joblib==1.5.1

kiwisolver==1.4.8

matplotlib==3.10.3

nonconformist==2.1.0

numpy==2.3.1

packaging==25.0

pandas==2.3.0

pillow==11.2.1

pyparsing==3.2.3

python-dateutil==2.9.0.post0

pytz==2025.2

scikit-learn==1.6.1

scipy==1.16.0

seaborn==0.13.2

six==1.17.0

sklearn-compat==0.1.3

threadpoolctl==3.6.0

tzdata==2025.2

xgboost==3.0.2

## 7.Methodology
### 7.1 Data Processing
This study applied variance filtering to remove low-variance features and used SIS to eliminate features with low correlation to the target variable. (Note: For datasets where the target variable is in the first row or the last column, the dataset with the target variable in the first row was transposed using `df.transpose()`, while datasets with the target variable in the last column were manually adjusted to place the target variable in the first column.) This approach helps reduce the complexity of the experimental section.

### 7.2 Modeling and Evaluation

Three classifiers—Random Forest, Support Vector Machine (SVM), and XGBoost—were trained on the eight datasets after variance filtering and feature selection.

Stratified training-test splits and five-fold cross-validation were used for evaluation. The evaluation metrics included accuracy, F1 score, Area Under the Receiver Operating Characteristic Curve (ROC-AUC), average precision, and G-mean.

To quantify prediction uncertainty, Split Conformal Prediction (SCP) was introduced, and the trade-off between coverage and prediction set size at different confidence levels was analyzed.

## 8. Citations
In this study, we utilized the icgc LUAD dataset for lung cancer prediction. The gene expression data from this dataset was used to train and evaluate deep learning models. Specifically, Liu S and Yao W (2022) proposed a deep learning method with KL divergence gene selection based on gene expression to improve the predictive accuracy of lung cancer.

# Reference

Liu S, Yao W. Prediction of lung cancer using gene expression and deep learning with KL divergence gene selection. BMC Bioinformatics, 2022, 23(1): 175.

## 1. Materials and Methods
### Computational Infrastructure:
python python 3.12 Platform: x86_64-w64-mingw32/x64 (64-bit) Running under: Windows >= 10 x64 (build 26100) The packages loaded:numpy_1.19.0, pandas_1.1.0, scikit-learn_0.24.0,xgboost_1.3.0, imbalanced-learn_0.8.0, matplotlib_3.3.0, seaborn_0.11.0
 ## 1.Conclusions
 ### Limitations
Although the method has shown good results on lung cancer gene expression datasets, its applicability to other types of high-dimensional imbalanced datasets (such as other cancer types or non-genetic datasets) still needs further verification. Therefore, the generalizability and cross-domain effectiveness of the ESCP method require more experimental validation.




