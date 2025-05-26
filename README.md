# ADHD_ML

## About dataset:

HYPERAKTIV is a public dataset containing health, activity, andheart rate data from adult patients 
diagnosed with attention deficit hyperactivity disorder, better known as ADHD. The dataset consists 
of data collected from 51 patients with ADHD and 52 controls. Data includes activity 
and heart rate data, series of patient attributes such as their age, sex, and 
information about their mental state, as well as output data from a computerized neuropsychological test.

## Data overview

Features data: Pre-extracted features from activity data  
Patient info: 32 Column in section 4  
HRV data: Each patient is separated to its own file. Heart rate variability data  
Activity data: Each patient is separated to its own file. Each file contains actvitiy metadata and measurement  
CPT II data: Contains individual responses to 360 CPT-II test trials, omission/commission errors, and the ADHD Confidence Index.

## Potential research goals and designs

Paper suggests the following questions:

* Predict whether a patient has ADHD or not by using the included activity data, heart rate variability, or a combination of the two.  
* Use patient-related attributes to analyze associations between ADHD and other illnesses such as bipolar disorder.
* Use patient-related attributes and unsupervised techniques to gain new insight to potentially advance diagnosis and treatment of ADHD and related mental disorders.
* Analyze the heart rate data in context to ADHD.

As highlighted by Loh et al., there are few AI studies that have attempted to use wearable technology for ADHD diagnosis
This gap presents a unique opportunity to explore question 1. proposed by Hicks et al.
The baseline models provided in the original paper offer a foundation for comparing traditional machine learning approaches with more advanced deep learning models.

To incorporate additional data types beyond wearable technology, such as CPT and information, we could also
implement dimensionality reduction techniques to identify the most predictive features.

## Methods
To ensure robust and unbiased evaluation, 10-fold cross-validation (10-CV) will be implemented across all modeling paths. 10 CV is 
the recommended value based on literature gathering from Loh et al.

Four possible outlines:  
* Focus on a subset of the data, such as activity or HRV data. Implement CNN to predict whether patient has ADHD or not
* Utilizing full dataset, employ dimensionality reduction. Implement CNN to predict whether patient has ADHD or not
* Utilizing full dataset, employ subnetworks for prediction. Combine output and use majority voting to predict whether patient has ADHD or not
* Utilizing full dataset, employ subnetworks for prediction. Feed the combined representation into a fully connected neural network for the final classification.

##### Feature Selection and Dimensionality Reduction
Use techniques like Akaike Information Criterion (AIC) to select prominent features from the dataset

##### Subnetwork approach
Given the heterogeneity of data types, a modular approach will be adopted where each data type is processed independently before integration.
This would leverage the cloud system design proposed by Loh et al., where each type of data could potentially have its own subnetwork. 
We would then employ a majority voting mechanism to aggregate the outputs from various subnetworks, determining whether a person is classified as having ADHD or not.

##### Potential models for buidling subnetworks include:
CNN signmoid activation: To process features extracted from time-series data from wearable devices
GNN: For patient attributes

##### Validation
Model performance will be evaluated on accuracy, precision, F1-score, recall, and MCC


## References

* Hicks, S. A., Stautland, A., Ole Bernt Fasmer, Wenche Førland, Hugo Lewi Hammer, Halvorsen, P., Mjeldheim, K., Ketil Joachim Oedegaard, Osnes, B., Giæver, E., Michael Alexander Riegler, & Jakobsen, P. (2021). HYPERAKTIV. https://doi.org/10.1145/3458305.3478454
* Loh, H. W., Ooi, C. P., Barua, P. D., Palmer, E. E., Molinari, F., & Acharya, U. R. (2022). Automated detection of ADHD: Current trends and future perspective. Computers in Biology and Medicine, 146, 105525. https://doi.org/10.1016/j.compbiomed.2022.105525

‌