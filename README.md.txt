# Customer Churn Prediction with CLTV-Based Segmentation

## Project Overview

In the telecommunications sector, retaining existing customers is significantly more profitable than acquiring new ones. This project builds an end-to-end machine learning pipeline to predict customer churn and combine churn risk with Customer Lifetime Value (CLTV) so that the business can prioritize profitable high-risk customers for retention.

The analysis answers two main questions:

1. Which customers have very high / high / medium / low / very low churn risk?
2. Among high-risk customers, which are financially worth retaining based on CLTV?

## Dataset

- Source: cleaned and enriched Telco Customer Churn dataset (Kaggle).
- Size: 7,043 customers and 52 features, including demographics, services, billing, CLTV, and churn labels.
- Target: `churn_rate` (binary churn indicator).

Place `Telco_customer_churn_cleaned.csv` in the `data/` folder.

## Methodology

1. **Data Preparation**  
   - Import libraries and load the dataset from `data/Telco_customer_churn_cleaned.csv`.[file:78]  
   - Basic cleaning: remove duplicate rows, check missing values, handle monetary fields, and standardize categorical encodings.[file:78]

2. **Feature Engineering**  
   - Split variables into numeric and categorical groups.[file:78]  
   - One-hot encode categorical features using `pd.get_dummies`.
   - Scale numeric features (`tenure`, `MonthlyCharges`, `TotalCharges`, `CLTV`, `Total Revenue`) with `StandardScaler`.[file:78]  
   - Concatenate all engineered variables into a 49-dimensional feature matrix.

3. **Modeling**  
   - Train/test split: 80/20 with stratification on churn.
   - Neural network (Keras Sequential):
     - Input: 49 features.
     - Hidden layers: 64 and 8 units with ReLU and dropout.
     - Output: sigmoid neuron for churn probability.
   - Classical ML models:
     - RandomForest, Naive Bayes, KNN, SVM, ExtraTrees, AdaBoost, GradientBoosting, CatBoost.

4. **Evaluation**  
   - Neural network: training and test accuracy, ROC curve, ROC-AUC. 
   - Classical models: 10-fold cross-validation with mean ROC-AUC and accuracy plus standard deviation.

5. **Churn Rate Categories and CLTV Segmentation**  
   - Convert predicted probabilities into churn risk bands: very high, high, medium, low, very low. 
   - Use CLTV distribution to define profitable customers (above 20th percentile).
   - Flag customers with churn probability ≥ 60% and CLTV ≥ 20th percentile as action candidates for targeted retention.

## Results

- Neural network:
  - Test accuracy ≈ 95.6%.
  - ROC-AUC ≈ 0.99.

- Best classical models (CatBoost, GradientBoosting, AdaBoost, RandomForest):
  - ROC-AUC ≈ 0.98–0.99, accuracy ≈ 94–95%.

Both the neural network and CatBoost deliver strong performance; the neural network is used as the main model for segmentation.

## Business Insights

- Focus retention spend on customers who are both high churn risk and high CLTV, instead of treating all churners equally.
- Use churn scores to trigger targeted offers and outreach at key moments in the customer journey.
- Use churn and CLTV patterns to refine product bundles and customer experience for vulnerable segments.

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/customer-churn-prediction.git
cd customer-churn-prediction
