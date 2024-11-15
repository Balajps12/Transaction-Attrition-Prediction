# Transaction Card Attrition Prediction

## Abstract
Transaction card attrition is a persistent challenge faced by financial institutions, where cardholders discontinue using their credit or debit cards. This phenomenon not only leads to revenue loss but also erodes customer relationships. This project aims to develop a predictive model to forecast transaction card attrition, helping financial institutions retain customers and enhance overall loyalty. The result is improved profitability for transaction card portfolios, strengthening the financial health of these institutions. This project serves as a proactive solution to an industry-wide challenge by leveraging data and machine learning to equip the banking sector with tools to predict and mitigate customer churn.

---

## Introduction
In today's data-driven business landscape, understanding and optimizing customer retention is key to growth and profitability. With the surge in non-cash electronic transactions, there’s been a rise in the use of debit and credit card transactions. Banks now face intense competition, with consumer behavior and loyalty at the heart of this dynamic industry. Since long-term clients are linked to greater profitability, banks benefit from reducing customer attrition to maintain steady growth.

## Literature Review
Research on customer attrition, especially in banking, emphasizes the high cost of customer loss. Studies across sectors like telecom and e-commerce reveal insights into churn prediction accuracy, customer behavior, and factors influencing churn:
1. Neslin et al. highlighted the significance of methodological accuracy in churn prediction models [1].
2. Ahn et al. examined how call quality impacts customer churn in telecom [2].
3. Hadden et al. compared various churn prediction systems with a focus on churn control accuracy [3].
4. Richter et al. identified that social interactions play a role in improving churn prediction in mobile networks [4].
5. Jahromi et al. developed a data mining technique to identify churn likelihood in a B2B context [5].

While previous work has largely focused on churn prediction, this project examines specific traits and variables contributing to customer churn in banking.

## Data Collection
The dataset, "Credit Card Attrition Rate Prediction," was sourced from Kaggle. It includes 17,000 records and 11 features, divided into predictor and target variables. Predictor variables provide customer details, while the target variable distinguishes between "Attrited" and "Existing" customers.

## Methodology
This project applies logistic regression and additional machine learning techniques to predict cardholder churn. Steps include:
1. **Data Collection and Preprocessing**: Cleaning and structuring historical data to ensure consistency.
2. **Feature Engineering**: Extracting spending patterns and demographic data.
3. **Model Development**: Creating a baseline logistic regression model, then testing additional models (e.g., decision trees, random forests, gradient boosting) for improved accuracy.
4. **Model Evaluation**: Dividing data into training and testing groups to assess accuracy, precision, recall, and F1-score.
5. **Model Optimization**: Selecting the best-performing model and optimizing it through hyperparameter tuning.

Using this approach, the model identifies customers at risk of attrition, enabling proactive retention strategies for financial institutions.

### Data Preprocessing
Data preprocessing includes:
- **Exploratory Data Analysis (EDA)**: Visually and statistically examining data to ensure quality.
- **Data Cleaning**: Addressing missing values and outliers.
- **Feature Scaling**: Normalizing numerical variables for comparability.
- **Encoding Categorical Variables**: One-hot encoding categorical variables for machine learning compatibility.

### Logistic Regression
Logistic Regression forms the core of this project's predictive model. It models the probability of attrition based on past transactions and customer data. As a binary classification algorithm, it identifies likely attritors versus loyal customers. With its interpretability and handling of various feature types, logistic regression serves as an essential baseline model.

Logistic Regression formula:
\[
\pi(X) = \frac{e^{\beta_0 + \beta_1 X_1 + \dots + \beta_k X_k}}{1 + e^{\beta_0 + \beta_1 X_1 + \dots + \beta_k X_k}}
\]

**Accuracy**: Logistic regression achieved 79.5% accuracy on training data and 80% accuracy on testing data.

## Implementation
A Streamlit web app is integrated to allow users to input features like Gender, Marital Status, Card Category, Dependent Count, etc., to predict whether a customer will continue using the card or attrite. Figures below display the Streamlit interface and output samples.

## Future Scope
The predictive model for transaction card attrition offers significant potential beyond financial services. By leveraging real-time data and refined machine learning methods, the model can drive tailored retention strategies that promote long-term loyalty across industries. Further enhancements, such as feedback-driven updates and advanced data collection, can solidify its value in customer relationship management.

---

## References
1. Neslin, S.A., et al., "Defection detection: Measuring and understanding the predictive accuracy of customer churn models," Journal of Marketing Research, 2006.
2. Ahn, J.H., et al., "Customer churn analysis: Churn determinants and mediation effects of partial defection," Telecommunications Policy, 2006.
3. Hadden, J., et al., "Computer assisted customer churn management: State-of-the-art and future trends," Computers & Operations Research, 2007.
4. Richter, Y., et al., "Predicting customer churn in mobile networks through analysis of social groups," SIAM Int'l Conference on Data Mining, 2010.
5. Jahromi, A.T., et al., "Managing B2B customer churn, retention and profitability," Industrial Marketing Management, 2014.
6. Guliyev, H., & Tatoğlu, F.Y., "Customer churn analysis in banking sector: Evidence from explainable machine learning models," Journal Of Applied Microeconometrics, 2021.
7. Chen, J.I., & Lai, K.L., "Deep convolution neural network model for credit-card fraud detection and alert," Journal of Artificial Intelligence, 2021.
8. Nie, G., et al., "Credit card churn forecasting by logistic regression and decision tree," Expert Systems with Applications, 2011.
9. Zhang, W., "Bank Customer Churn Analysis and Prediction," MSIEID 2022 Conference Proceedings, 2023.
10. Hussein, A.S., et al., "Credit Card Fraud Detection Using Fuzzy Rough Nearest Neighbour," Int'l Journal of Interactive Mobile Technologies, 2021.
11. Mittal, S., & Tyagi, S., "Performance evaluation of machine learning algorithms for credit card fraud detection," IEEE Confluence, 2019.

---
