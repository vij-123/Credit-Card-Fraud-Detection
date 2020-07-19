# Credit-Card-Fraud-Detection

Problem Statement:
The Credit Card Fraud Detection Problem includes modeling past credit card transactions with the knowledge of the ones that turned out to be fraud. This model is then used to identify whether a new transaction is fraudulent or not.

For Dataset:-
https://www.kaggle.com/mlg-ulb/creditcardfraud

Dataset:-
The data contains 284,807 transactions that occurred over a two-day period, of which 492 (0.17%) are fraudulent. Each transaction has 30 features, all of which are numerical. The features V1, V2, ..., V28 are the result of a PCA transformation. To protect confidentiality, background information on these features is not available. The Time feature contains the time elapsed since the first transaction, and the Amount feature contains the transaction amount. The response variable, Class, is 1 in the case of fraud, and 0 otherwise.

Data Set Analysis:
1. The data set is highly skewed, consisting of 492 frauds in a total of 284,807 observations. This resulted in only 0.172% fraud cases. This skewed set is justified by the low      number of fraudulent transactions.
2. The dataset consists of numerical values from the 28 ‘Principal Component Analysis (PCA)’ transformed features, namely V1 to V28. Furthermore, there is no metadata about the      original features provided, so pre-analysis or feature study could not be done.
3. The ‘Time’ and ‘Amount’ features are not transformed data.
4. There is no missing value in the dataset.

Algorithms used:-
This problem is a classification problem so i have used classification algorithms.
we will see which algo works best for this problem by judging accuracy:
1. Logistic regression
2. Random Forest Classifier  

Result:-
we got a 99.95% accuracy from random forest and 99.92% accuracy from logistic regression.
so random forest works best for this classification problem.
