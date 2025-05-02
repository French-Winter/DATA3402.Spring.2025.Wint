# Marketing Campaign Data Set - Binary Classification
## Objective was to create a binary classification *response model which can provide a significant boost to the efficiency of a marketing campaign by increasing responses or reducing expenses.*

### Initial Overview:

- *AcceptedCmp1 - 1 if customer accepted the offer in the 1st campaign, 0 otherwise*
- *AcceptedCmp2 - 1 if customer accepted the offer in the 2nd campaign, 0 otherwise*
- *AcceptedCmp3 - 1 if customer accepted the offer in the 3rd campaign, 0 otherwise*
- *AcceptedCmp4 - 1 if customer accepted the offer in the 4th campaign, 0 otherwise*
- *AcceptedCmp5 - 1 if customer accepted the offer in the 5th campaign, 0 otherwise*
- ***Response (TARGET) - 1 if customer accepted the offer in the last campaign, 0 otherwise***
- *Complain - 1 if customer complained in the last 2 years*
- *DtCustomer - date of customer’s enrolment with the company*
- *Education - customer’s level of education*
- *Marital - customer’s marital status*
- *Kidhome - number of small children in customer’s household*
- *Teenhome - number of teenagers in customer’s household*
- *Income - customer’s yearly household income*
- *MntFishProducts - amount spent on fish products in the last 2 years*
- *MntMeatProducts - amount spent on meat products in the last 2 years*
- *MntFruits - amount spent on fruits products in the last 2 years*
- *MntSweetProducts - amount spent on sweet products in the last 2 years*
- *MntWines - amount spent on wine products in the last 2 years*
- *MntGoldProds - amount spent on gold products in the last 2 years*
- *NumDealsPurchases - number of purchases made with discount*
- *NumCatalogPurchases - number of purchases made using catalogue*
- *NumStorePurchases - number of purchases made directly in stores*
- *NumWebPurchases - number of purchases made through company’s web site*
- *NumWebVisitsMonth - number of visits to company’s web site in the last month*
- *Recency - number of days since the last purchase*

(All italicized notes were taken from the Kaggle.com under the project description for this dataset)

### User Instructions

Assuming you already have a working Linux terminal (Windows systems) or ... (Apple systems), and already have python3 installed, you will need to install the following packages:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn

You can instal these using the following command on your terminal:
- pip install python3-xyz

Where 'xyz' will be the name of the packaged you mean to install. 

### Data Exploration

Once you've downloaded the data set from Kaggle, uploaded it to Jupyter Notebook, and downloaded the required packages for the code, explore. 

Upon first looks, the data set is 2240 rows (entries) by 29 columns (features) with only missing values in the 'Income' column. 

All columns are numeric except 'Education' and 'Marital_Status' which are categorical, also 'Dt_Customer' which should be data-time. Also, 'Z_CostContact' and 'Z_Revenue' are constants. 

There are many outlier values in the data set but the only notable outliers are in the 'Year_Birth' column.

The target column, 'Response', is already encoded 'True' and 'False' to '1' and '0'. However, the target has an imbalance of 0's and 1's. The class imbalance is visualized below.

### Basic Data Visuals
Bar charts of initial features:
![image](https://github.com/user-attachments/assets/979223fb-a878-4ec9-bbe9-b850b56dadba)

Correlation Heatmap of initial features:
![image](https://github.com/user-attachments/assets/fc842a5d-d403-4683-aec2-0ffa6b7e12f2)

Bar chart of class imbalance in 'Response' column:
![image](https://github.com/user-attachments/assets/4af91947-8ae9-4ef3-a0f2-6b7a129c4261)

### How it Works

The data set is cleaned and preprocessed before the model could be trained.

**Data Cleaning:**
- Missing values in 'Income' column were imputed with zero.
- Outlier values in 'Year_Birth' were imputed with '1925'.
- Later on: Infinite values were imputed with zero.

**Data Standardization, Feature Engineering, and Encoding:**
- All numeric columns were standardized using scikit-learn's standard scalar.
- 'Marital_Status' categories which valued less than 1% of the data were consolidated into 1 category.
- From 'Year_Birth' the customer age was extracted, placed into a new column 'Age' then seperated customers into different age groups.
- From 'Dt_Costumer' the customer subscription length was extracted into 'Customer_Years' column.
- 'MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts', and 'MntGoldProducts' were summed and extracted to 'TotalSpend' column.
- From 'TotalSpend' and 'Income' their ratio was extracted to 'SpendToIncomeRatio' column.
- 'Education' and 'Marital_Status' were one-hot encoded.
- Ratios of other highly correlated features were extracted into new columns.

**Model Training & Selection:**
- The data set was split into training and testing subsets. The columns 'ID', 'Education', 'Marital_Status', 'Dt_Customer', 'Z_CostContact', 'Z_Revenue', and 'Response' were dropped from the features set before training and testing begun. 'Response' column was stored in the seperate variable.
- After data was split into training and testing subsets, Synthetic Minority Oversampling Technique (SMOTE) was applied to correct the class imbalance in the target column.
- Logistic Regression, Random Forest, SVM, and KNN were the selected models. Accuracy, Precision, Recall, F1-Score, were the selected metrics. 
- After running the models, the metrics returned that the Random Forest model showed the best performance on all metrics.

  
=== Model Comparison ===

**Logistic Regression**

Accuracy:  0.8065
Precision: 0.4148
Recall:    0.7300
F1-Score:  0.5290

**Random Forest**

Accuracy:  0.8839
Precision: 0.6618
Recall:    0.4500
F1-Score:  0.5357

**SVM**

Accuracy:  0.8318
Precision: 0.4532
Recall:    0.6300
F1-Score:  0.5272

**KNN**

Accuracy:  0.7872
Precision: 0.3799
Recall:    0.6800
F1-Score:  0.4875


- The ROC Curve of the Random Forest Model returned an AUC of 0.87, which is less than before the data set handling but since all other metrics improved, we will accept this.

![image](https://github.com/user-attachments/assets/2d221298-a027-4e17-a9ae-fef0a6ce46f3)

### Conclusion

The data set presented few problems in terms of data cleaning. However, there was much work to be done in dealing with multicollinearity issues and target class imbalance. 

Overall, the 'AcceptedCmp{num}', 'NumCatalogPurchases', and 'SpendToIncomeRatio' features had the highest relation to the 'Response' column and therefore were the most helpful in the binary classification. 

Correcting the class imbalance also allowed the model to be more familiar with the preferred outcome of the target column.

### Future Goals

In the future for this project, I intend to:
- Further improve the metrics of the Random Forest model.
- Correct the class imbalance using other oversampling techniques (Adasin, for example).
- Use more complex models and measure their performance to Random Forest.

