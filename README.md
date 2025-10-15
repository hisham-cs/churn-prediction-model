# Customer Churn Prediction using Logistic Regression

This project focuses on predicting customer churn for a telecommunications company. The main goal is to build a machine learning model that can identify customers who are likely to cancel their service. By identifying these at-risk customers, the business can take proactive measures to retain them.

## 📊 Dataset

The dataset used is the popular **Telco Customer Churn** dataset, which contains information about:
* Customer demographics (gender, age range, and dependents).
* Services they have signed up for (phone, multiple lines, internet, etc.).
* Account information (how long they've been a customer, contract, payment method).
* The target variable, `Churn`, indicating whether the customer left within the last month.

## ⚙️ Project Workflow

The project follows a standard machine learning workflow:
1.  **Data Loading and Cleaning:** The dataset was loaded, and initial cleaning was performed, which included handling missing values, correcting data types, and dropping irrelevant columns like `customerID`.
2.  **Feature Engineering:** Categorical variables were converted into a numerical format using `LabelEncoder` for binary features and `OneHotEncoder` for multi-class features.
3.  **Data Splitting and Scaling:** The data was split into training and testing sets. Features were then scaled using `StandardScaler` to ensure all variables were on a similar scale.
4.  **Model Training:** A **Logistic Regression** model was trained on the preprocessed data.
5.  **Evaluation:** The model's performance was assessed using key classification metrics.

## 📈 Results & Key Findings

A crucial part of this project was understanding the impact of **class imbalance** on model performance.

### Model 1: Before Balancing

The initial model achieved a seemingly high accuracy of **~81%**. However, it performed poorly on the most important task: identifying customers who would actually churn. Its **Recall** was only **55%**, meaning it missed almost half of the at-risk customers.

<img width="563" height="455" alt="download (1)" src="https://github.com/user-attachments/assets/8f9fb814-96eb-474f-ba6e-e12dd52f7cda" />


### Model 2: After Balancing

To solve this, the `class_weight='balanced'` parameter was used in the Logistic Regression model. This tells the model to pay more attention to the minority class (churners). The results were much more useful for a real-world business scenario:

* The overall accuracy dropped slightly to **~75%**.
* Crucially, the **Recall** score jumped from **55% to 81%**!

This shows that the improved model is now able to correctly identify the vast majority of customers who are likely to churn.

<img width="563" height="455" alt="download (2)" src="https://github.com/user-attachments/assets/3ffe6ace-581b-4eec-af80-49ea5cc17e14" />


## 💡 Conclusion

The main takeaway from this project is that **accuracy can be a misleading metric**, especially with imbalanced datasets. For business problems like churn prediction, focusing on metrics like **Recall** is far more important. A model's success should be measured by its ability to provide actionable insights and solve a real-world problem, not just by a high accuracy score.

## 🚀 How to Run

To run this project yourself:
1.  Clone the repository.
2.  Ensure you have Python and the required libraries installed:
    ```bash
    pip install pandas numpy scikit-learn matplotlib
    ```
3.  Place the `Telco-Customer-Churn.csv` file in the same directory.
4.  Run the Jupyter Notebook.
