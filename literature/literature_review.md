## Title

# Heart Disease Prediction Using Machine Learning

---

# Part A — Identification and Collection of a Suitable Dataset

## 1. Introduction

Heart disease is an important healthcare problem, and Machine Learning (ML) techniques can be used to analyze clinical and demographic data for predictive classification. The objective of this study is to investigate the application of Machine Learning algorithms for predicting the presence of heart disease.

For this project, the **UCI Heart Disease dataset**, specifically the **Cleveland subset**, has been selected as the primary dataset. The dataset is available through the UCI Machine Learning Repository and is categorized as a multivariate classification dataset in the health and medicine domain. The UCI repository identifies **303 instances** and **13 predictor features** for the commonly used Cleveland dataset.

---

## 2. Dataset Source

| Attribute | Details |
|---|---|
| **Dataset** | Heart Disease — Cleveland subset |
| **Source** | UCI Machine Learning Repository |
| **Dataset ID** | 45 |
| **Original researchers** | Janosi, Steinbrunn, Pfisterer, and Detrano |
| **Year** | 1989 |
| **DOI** | 10.24432/C52P4X |
| **Number of instances** | 303 |
| **Number of predictor features** | 13 |
| **Problem type** | Classification |
| **Domain** | Health and Medicine |

The UCI repository contains four databases associated with the Heart Disease dataset:

1. Cleveland
2. Hungary
3. Switzerland
4. VA Long Beach

UCI notes that the **Cleveland database** is the one that has been used in published Machine Learning experiments. The published experiments generally use a subset of 14 attributes, consisting of **13 predictors and the diagnosis attribute**.

---

## 3. Dataset Features

The selected dataset contains the following predictor variables:

### Patient Features and Descriptions

| No. | Feature | Description |
|---:|---|---|
| 1 | `age` | Age of the patient in years |
| 2 | `sex` | Sex of the patient |
| 3 | `cp` | Chest-pain type |
| 4 | `trestbps` | Resting blood pressure |
| 5 | `chol` | Serum cholesterol |
| 6 | `fbs` | Fasting blood sugar > 120 mg/dl |
| 7 | `restecg` | Resting electrocardiographic result |
| 8 | `thalach` | Maximum heart rate achieved |
| 9 | `exang` | Exercise-induced angina |
| 10 | `oldpeak` | ST depression induced by exercise |
| 11 | `slope` | Slope of peak exercise ST segment |
| 12 | `ca` | Number of major vessels observed by fluoroscopy |
| 13 | `thal` | Thalassemia result |

These feature definitions are provided in the official UCI dataset documentation.

---

## 4. Target Variable

The original target variable is `num`.

It contains values from **0 to 4**:

- `0` — Absence of heart disease
- `1–4` — Presence of heart disease

The UCI documentation states that experiments using the Cleveland database have generally focused on distinguishing **absence of disease (0)** from **presence of disease (1–4)**.

Therefore, for this project, the target will be converted into a **binary classification variable**.

### Target Conversion

| Original `num` | Project Target | Meaning |
|---:|---:|---|
| 0 | 0 | No heart disease |
| 1 | 1 | Heart disease |
| 2 | 1 | Heart disease |
| 3 | 1 | Heart disease |
| 4 | 1 | Heart disease |

Thus, the Machine Learning problem can be expressed as:

> Given a patient's demographic and clinical attributes, classify whether the patient belongs to the absence-of-heart-disease or presence-of-heart-disease class.

---

## 5. Dataset Suitability

The selected dataset is suitable for this project for several reasons.

First, it directly represents a **binary classification problem**, which is appropriate for applying and comparing supervised Machine Learning algorithms.

Second, it contains a combination of **numerical and categorical attributes**. This allows the project to demonstrate important Machine Learning preprocessing operations such as:

- Categorical encoding
- Feature scaling
- Missing-value handling

Third, the dataset has been extensively used in previous Machine Learning research. The UCI repository currently lists numerous papers citing the dataset and specifically notes its use in published ML experiments.

Finally, the dataset's extensive use in previous research makes it particularly appropriate for the second component of this assignment: **literature review and comparison with previous research**.

---

## 6. Planned Data Analysis

The dataset will subsequently be examined using **Exploratory Data Analysis (EDA)**.

The analysis will investigate:

- Missing values
- Duplicate records
- Data types
- Feature distributions
- Class distribution
- Outliers
- Correlations between numerical features
- Relationships between important features and the target
- Potential class imbalance

Appropriate preprocessing techniques will then be applied before Machine Learning model development.

---

## 7. Dataset Citation

The dataset will be cited as:

> Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease*. UCI Machine Learning Repository. DOI: 10.24432/C52P4X.

---

# Part B — Literature Review

## 8. Introduction to the Literature Review

Machine Learning has been widely investigated for heart-disease prediction. Researchers have applied:

- Traditional classification algorithms
- Ensemble learning
- Feature-selection techniques
- Hybrid models
- Fuzzy systems
- Deep-learning methods

A systematic review published in 2024 identified **61 published studies** that compared multiple supervised Machine Learning algorithms for heart-disease prediction. The review found that Decision Trees and Random Forests were among the most frequently used algorithms and that researchers used a variety of datasets and evaluation metrics.

The present literature review therefore focuses on research relevant to the proposed **Heart Disease Prediction** project, with particular attention to:

- Studies involving the UCI/Cleveland dataset
- Comparative Machine Learning experiments
- Feature selection
- Ensemble methods
- Advanced ML approaches

---

# 9. Review of 20 Research Papers

## 9.1 Palaniappan and Awang (2008)

### Paper

**Intelligent Heart Disease Prediction System Using Data Mining Techniques**

Palaniappan and Awang developed an Intelligent Heart Disease Prediction System (IHDPS) using data-mining techniques. The study investigated Decision Trees, Naive Bayes, and Neural Networks for heart-disease prediction. The work represents an early application of data-mining methods to automated clinical decision support.

**Relevance:** Establishes an early foundation for applying supervised learning to heart-disease prediction.

---

## 9.2 Sultana, Haider and Uddin (2016)

### Paper

**Analysis of Data Mining Techniques for Heart Disease Prediction**

The authors evaluated KStar, J48, SMO, Bayes Net, and Multilayer Perceptron using Weka. Performance was evaluated using predictive accuracy, ROC curves, and AUC. SMO and Bayes Net showed strong performance in their experiments.

**Relevance:** Demonstrates the value of comparing several classification algorithms and multiple evaluation measures.

---

## 9.3 Abdar et al. (2015)

### Paper

**Predicting Heart Disease Using Machine Learning/Data Mining Techniques**

Abdar and colleagues investigated multiple classification approaches for heart-disease prediction, including:

- Decision Tree
- Neural Network
- SVM
- Logistic Regression
- KNN

**Relevance:** Provides a comparative baseline involving several algorithms that can also be investigated in the proposed project.

---

## 9.4 Arabasadi et al. (2017)

### Paper

**Computer Aided Decision Making for Heart Disease Detection Using Hybrid Neural Network-Genetic Algorithm**

The researchers proposed a hybrid approach combining a Genetic Algorithm with a Neural Network. Their experiments used the Z-Alizadeh Sani coronary artery disease dataset containing **303 patients and 54 features**.

The proposed method achieved reported:

- Accuracy: **93.85%**
- Sensitivity: **97%**
- Specificity: **92%**

**Relevance:** Demonstrates how optimization techniques can be combined with neural networks.

**Limitation for direct comparison:** The dataset is different from our Cleveland dataset.

---

## 9.5 Uyar and Ilhan (2017)

### Paper

**Diagnosis of Heart Disease Using Genetic Algorithm Based Trained Recurrent Fuzzy Neural Networks**

Uyar and Ilhan developed a Genetic Algorithm-trained Recurrent Fuzzy Neural Network using the UCI Cleveland dataset.

The study used:

- 297 available instances
- 252 instances for training
- 45 instances for testing

The study reported **97.78% testing accuracy**.

Other reported metrics included:

- Sensitivity
- Specificity
- Precision
- F-score
- Misclassification error

**Relevance:** Directly relevant because the Cleveland dataset is used.

---

## 9.6 Pahwa and Kumar (2017)

### Paper

**Prediction of Heart Disease Using Hybrid Technique for Selecting Features**

Pahwa and Kumar investigated feature-selection techniques together with classification methods, including Naive Bayes and Random Forest.

The study focused on reducing irrelevant or redundant attributes through a hybrid feature-selection approach.

**Relevance:** Feature selection is a possible extension of our experimental methodology.

---

## 9.7 Sharma and Saxena (2017)

### Paper

**Application of Fuzzy Logic and Genetic Algorithm in Heart Disease Risk Level Prediction**

The authors proposed a fuzzy-rule-based clinical decision-support system combined with Genetic Algorithms for heart-disease risk prediction.

The study focused on:

- Generating weighted fuzzy rules
- Identifying risk levels
- Using clinical attributes

**Relevance:** Shows an alternative to conventional ML classification through fuzzy and evolutionary approaches.

---

## 9.8 Haq et al. (2018)

### Paper

**A Hybrid Intelligent System Framework for the Prediction of Heart Disease Using Machine Learning Algorithms**

Haq et al. evaluated seven classifiers:

1. Logistic Regression
2. KNN
3. ANN
4. SVM
5. Naive Bayes
6. Decision Tree
7. Random Forest

They also investigated three feature-selection methods:

- Relief
- mRMR
- LASSO

The Cleveland heart-disease dataset was used, and k-fold cross-validation was applied.

Their study reported that feature selection could affect classification accuracy and computational time.

**Relevance:** Highly relevant to our project because it uses the Cleveland dataset and combines feature selection with comparative ML.

---

## 9.9 Dwivedi (2018)

### Paper

**Performance Evaluation of Different Machine Learning Techniques for Prediction of Heart Disease**

Dwivedi evaluated six Machine Learning techniques using several classification performance measures and ROC analysis.

Logistic Regression achieved the highest reported accuracy of **85%**, with:

- Sensitivity: **89%**
- Specificity: **81%**

**Relevance:** Demonstrates why our study should use several evaluation metrics rather than accuracy alone.

---

## 9.10 Amin, Chiam and Varathan (2019)

### Paper

**Identification of Significant Features and Data Mining Techniques in Predicting Heart Disease**

The authors evaluated seven classification algorithms and investigated significant features for heart-disease prediction.

The Cleveland dataset was selected from the UCI repository, and the researchers also used another dataset for validating their findings.

Their proposed model using significant features and a voting technique achieved a reported accuracy of **87.4%**.

**Relevance:** Directly relevant to our dataset and provides evidence for investigating feature importance and feature selection.

---

## 9.11 Mohan, Thirumalai and Srivastava (2019)

### Paper

**Effective Heart Disease Prediction Using Hybrid Machine Learning Techniques**

The authors proposed a hybrid Random Forest and Linear Model approach for heart-disease prediction.

The study investigated:

- Significant features
- Hybrid learning
- Predictive performance

**Relevance:** Provides a basis for investigating ensemble and hybrid approaches after establishing baseline models.

---

## 9.12 Latha and Jeeva (2019)

### Paper

**Improving the Accuracy of Prediction of Heart Disease Risk Based on Ensemble Classification Techniques**

This study investigated ensemble classification methods for improving the performance of weaker classifiers.

The study evaluated:

- Bagging
- Boosting

The authors reported improvements in prediction accuracy and also examined feature selection.

**Relevance:** Supports the investigation of ensemble methods such as Random Forest and boosting algorithms.

---

## 9.13 Ali et al. (2019)

### Paper

**An Automated Diagnostic System for Heart Disease Prediction Based on χ² Statistical Model and Optimally Configured Deep Neural Network**

Ali et al. proposed an automated diagnostic system using a χ² statistical feature-selection approach and an optimally configured Deep Neural Network.

The work represents a transition from conventional classifiers toward optimized deep-learning approaches.

**Relevance:** Provides a reference point for advanced feature-selection and deep-learning approaches.

---

## 9.14 Al-Makhadmeh and Tolba (2019)

### Paper

**Utilizing IoT Wearable Medical Device for Heart Disease Prediction Using Higher Order Boltzmann Model: A Classification Approach**

The researchers proposed an IoT-based medical monitoring system combined with a Higher Order Boltzmann Deep Belief Neural Network.

They evaluated the system using measures including:

- F-measure
- Sensitivity
- Specificity
- ROC analysis

They reported **99.03% accuracy** in their experimental setting.

**Relevance:** Demonstrates the integration of IoT data collection and deep learning.

**Limitation for direct comparison:** The experimental setting is substantially different from our 303-record Cleveland dataset.

---

## 9.15 Tougui, Jilbab and El Mhamdi (2020)

### Paper

**Heart Disease Classification Using Data Mining Tools and Machine Learning Techniques**

Tougui et al. compared six data-mining platforms and six ML algorithms:

- Logistic Regression
- SVM
- KNN
- ANN
- Naive Bayes
- Random Forest

Their dataset contained **303 instances and 13 features**.

Accuracy, sensitivity, and specificity were used as performance measures.

**Relevance:** This is one of the most directly comparable studies for our project because the dataset structure and major algorithms closely match our proposed experiment.

---

## 9.16 Khan et al. (2020)

### Paper

**Empirical Study of Various Classification Techniques for Heart Disease Prediction**

Khan and colleagues conducted an empirical comparison of classification techniques for heart-disease prediction.

The study is relevant because it investigates the comparative performance of different classifiers rather than relying on a single predictive method.

**Relevance:** Supports our decision to perform a comparative ML experiment.

---

## 9.17 Kavitha et al. (2021)

### Paper

**Heart Disease Prediction Using Hybrid Machine Learning Model**

Kavitha et al. proposed a hybrid model combining Decision Tree and Random Forest for heart-disease prediction using the Cleveland heart-disease dataset.

The reported hybrid model achieved **88.7% accuracy**.

**Relevance:** Particularly useful for comparison because it uses the Cleveland dataset and includes Random Forest and Decision Tree.

---

## 9.18 Bharti et al. (2021)

### Paper

**Prediction of Heart Disease Using a Combination of Machine Learning and Deep Learning**

Bharti et al. applied Machine Learning and deep-learning methods to the UCI Heart Disease dataset.

The researchers addressed irrelevant features using Isolation Forest and normalized the data.

Their deep-learning approach reported **94.2% accuracy**.

**Relevance:** Demonstrates the potential impact of preprocessing and advanced learning approaches.

---

## 9.19 Rani et al. (2021)

### Paper

**A Decision Support System for Heart Disease Prediction Based Upon Machine Learning**

Rani et al. investigated a Machine Learning-based decision-support system for heart-disease prediction.

The research emphasizes the potential role of automated predictive systems in supporting clinical decision-making, particularly in settings where specialist resources may be limited.

**Relevance:** Provides a decision-support perspective for the proposed project.

---

## 9.20 Teekaraman et al. (2022)

### Paper

**Implementation of a Heart Disease Risk Prediction Model Using Machine Learning**

Teekaraman et al. applied:

- SVM
- Gaussian Naive Bayes
- Logistic Regression
- LightGBM
- XGBoost
- Random Forest

to the Cleveland dataset.

The study used **303 records and 13 selected features**.

Random Forest achieved the highest reported validation accuracy in their experiment at **88.5%**.

**Relevance:** This is one of the most useful benchmark studies for our proposed experiment because it uses the same Cleveland dataset and several algorithms that we plan to investigate.

---

# 10. Comparative Literature Review

The 20 reviewed studies can be summarized as follows.

| No. | Study | Year | Main Approach | Dataset Relevance |
|---:|---|---:|---|---|
| 1 | Palaniappan & Awang | 2008 | DT, NB, NN | General heart disease |
| 2 | Sultana et al. | 2016 | KStar, J48, SMO, Bayes Net, MLP | Heart disease |
| 3 | Abdar et al. | 2015 | Multiple classifiers | Heart disease |
| 4 | Arabasadi et al. | 2017 | GA + NN | Z-Alizadeh Sani |
| 5 | Uyar & Ilhan | 2017 | GA + RFNN | Cleveland |
| 6 | Pahwa & Kumar | 2017 | Feature selection + ML | Heart disease |
| 7 | Sharma & Saxena | 2017 | Fuzzy + GA | Heart disease |
| 8 | Haq et al. | 2018 | Seven ML classifiers + FS | Cleveland |
| 9 | Dwivedi | 2018 | Six ML techniques | Heart disease |
| 10 | Amin et al. | 2019 | Feature selection + voting | Cleveland |
| 11 | Mohan et al. | 2019 | Hybrid ML | Heart disease |
| 12 | Latha & Jeeva | 2019 | Ensemble learning | Heart disease |
| 13 | Ali et al. | 2019 | χ² + DNN | Heart disease |
| 14 | Al-Makhadmeh & Tolba | 2019 | Deep learning + IoT | Different setting |
| 15 | Tougui et al. | 2020 | LR, SVM, KNN, ANN, NB, RF | 303-instance dataset |
| 16 | Khan et al. | 2020 | Comparative classifiers | Heart disease |
| 17 | Kavitha et al. | 2021 | DT + RF hybrid | Cleveland |
| 18 | Bharti et al. | 2021 | ML + deep learning | UCI |
| 19 | Rani et al. | 2021 | ML decision support | Heart disease |
| 20 | Teekaraman et al. | 2022 | SVM, NB, LR, LightGBM, XGBoost, RF | Cleveland |

---

# 11. Major Findings from the Literature

## 11.1 Comparative Machine Learning is Common

A recurring pattern in the literature is the comparison of several classification algorithms rather than the use of a single model.

The following algorithms occur repeatedly across the studies:

- Logistic Regression
- SVM
- KNN
- Decision Trees
- Naive Bayes
- Random Forest

The 2024 systematic review likewise found that many studies compared multiple supervised ML algorithms.

This supports the proposed methodology of establishing several baseline classifiers before considering more advanced models.

---

## 11.2 Random Forest and Ensemble Methods are Frequently Investigated

Random Forest and other ensemble techniques appear frequently in the literature.

Latha and Jeeva specifically investigated bagging and boosting, while Mohan et al. proposed a hybrid approach.

Teekaraman et al. also evaluated Random Forest alongside:

- SVM
- Naive Bayes
- Logistic Regression
- LightGBM
- XGBoost

on the Cleveland dataset.

---

## 11.3 Feature Selection is an Important Research Direction

Several studies investigate the selection of relevant features.

Haq et al. used:

- Relief
- mRMR
- LASSO

while Amin et al. specifically investigated significant features in the Cleveland dataset.

This suggests that the proposed project should investigate feature importance and possibly feature selection after establishing the baseline results.

---

## 11.4 Advanced and Hybrid Models are Increasingly Used

The literature includes:

- Genetic Algorithms
- Fuzzy neural networks
- Deep neural networks
- Hybrid classifiers
- Ensemble models

Uyar and Ilhan used a Genetic Algorithm-trained recurrent fuzzy neural network on the Cleveland dataset, while Bharti et al. investigated a combination of Machine Learning and deep learning.

---

## 11.5 Accuracy Alone is Not Sufficient

Different researchers use different evaluation metrics.

Examples include:

- Accuracy
- Sensitivity
- Specificity
- Precision
- Recall
- F1-score
- AUC/ROC
- Matthews correlation coefficient
- Confusion matrix

Dwivedi's study, for example, evaluated several classification measures and ROC analysis rather than reporting accuracy alone.

Therefore, the present project should report multiple evaluation metrics.

---

# 12. Research Gap

The literature review reveals several areas that can be investigated in the proposed study.

## Gap 1 — Lack of Uniform Experimental Conditions

Different studies use different:

- Train/test splits
- Cross-validation strategies
- Preprocessing methods
- Feature-selection methods
- Hyperparameters
- Evaluation metrics

Therefore, accuracy values reported by different papers cannot always be directly compared.

---

## Gap 2 — Dataset Differences

Some researchers use the Cleveland dataset, while others use different clinical datasets.

Consequently, an algorithm's reported performance on one dataset cannot automatically be assumed to transfer to another.

For example, the Cleveland dataset contains only **303 records**, while other studies use considerably larger clinical datasets.

---

## Gap 3 — Accuracy-Focused Evaluation

A substantial portion of the literature emphasizes accuracy.

A more complete evaluation using:

- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrices

can provide a broader understanding of model behaviour.

---

## Gap 4 — Reproducibility

Some studies use sophisticated hybrid or optimized approaches, but differences in preprocessing and validation make reproduction difficult.

The proposed study will therefore emphasize a clearly documented and reproducible pipeline.

---

## Gap 5 — Need for Controlled Comparison

A useful contribution for a student research project is to compare several algorithms under the same preprocessing, data split, and evaluation framework on the Cleveland dataset.

This does not claim that the resulting model will outperform every published method. Instead, it provides a controlled experimental comparison against selected published results.

---

# 13. Proposed Methodology Based on the Literature Review

Based on the literature findings, the proposed experimental workflow is:

```text
Dataset Collection
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Missing-Value Handling
        ↓
Categorical Encoding
        ↓
Feature Scaling where required
        ↓
Train/Test Split or Cross-Validation
        ↓
Baseline Models
        ├── Logistic Regression
        ├── KNN
        ├── Decision Tree
        ├── Naive Bayes
        ├── SVM
        └── Random Forest
        ↓
Advanced/Ensemble Model
        └── XGBoost or another justified ensemble method
        ↓
Performance Evaluation
        ├── Accuracy
        ├── Precision
        ├── Recall
        ├── F1-score
        ├── ROC-AUC
        └── Confusion Matrix
        ↓
Feature Importance / Interpretation
        ↓
Comparison with Selected Literature
        ↓
Discussion of Findings and Limitations
