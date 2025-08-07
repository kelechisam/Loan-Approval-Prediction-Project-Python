Project Overview

This project focuses on predicting loan approval status using machine learning models, including Logistic Regression, K-Nearest Neighbors (KNN), and Decision Tree classifiers. The dataset contains loan applicant details, and the goals are to identify key factors influencing loan approval and assess the predictive accuracy of each model.

Analysis Steps
# Data Loading and Initial Exploration

    Imported necessary libraries (pandas, numpy, matplotlib, seaborn, sklearn)
    
    Loaded the dataset and examined the first few rows
    
    Checked data types and basic statistics

# Data Understanding

    Verified no missing values in the dataset
    
    Checked for duplicate rows
    
    Examined distribution of numerical and categorical variables

# Exploratory Data Analysis (EDA)

    Visualized loan approval status distribution
    
    Analyzed income distribution by loan status
    
    Created histograms for all numerical features
    
    Examined relationships between variables

# Data Preprocessing

    Identified numerical and categorical columns for feature engineering
    
    Prepared data for machine learning modeling

# Model Building

    Set up preprocessing pipeline with OneHotEncoder and StandardScaler
    
    Prepared to use Logistic Regression for prediction
    
    Split data into training and test sets


#  import necessary library
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

#  Load CSV data
    df=pd.read_csv('/content/loan_data.csv')

#  Data Understanding
    df.info()
    df.describe()

#  Check for missing values
    print(df.isnull().sum())

#  Check for duplicate values
    print('number of duplicate rows:',df.duplicated().sum())

#  EDA(Exploratory Data Analysis)
    plt.figure(figsize=(12,6))
    sns.countplot(x='loan_status', data=df)
    plt.show()

    plt.figure(figsize=(12,6))
    sns.boxplot(x='loan_status',y='person_income', data=df)
    plt.title('Income Distribution by Loan Status')
    plt.show()

#  visualize for numerical features
    num_cols=['person_age','person_income','person_emp_exp','loan_amnt','loan_int_rate','loan_percent_income','cb_person_cred_hist_length','credit_score']
    df[num_cols].hist(bins=30, figsize=(15,10))
    plt.tight_layout()
    plt.show()

#  visualize for categorical features
    cat_cols= ['person_gender','person_education','person_home_ownership','loan_intent','previous_loan_defaults_on_file']
    for col in cat_cols:
    plt.figure(figsize=(10,4))
    sns.countplot(x=col,hue='loan_status', data=df)
    plt.title(f'Loan Status by {col}')
    plt.xticks(rotation=45)
    plt.show()

#  limit age to 100
    df=df[df['person_age'] < 100]

#  convert previous_default to yes =1 , no=0
    df['previous_default']=df['previous_loan_defaults_on_file'].map({'Yes':1, 'No':0})

#  target features
    features =['person_age','person_income','loan_amnt','loan_int_rate','credit_score','previous_default']
    X= df[features]
    Y= df['loan_status']

#  Split data into testing and training
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.2, random_state=42)

1. Logistic Regression Model

#  scaling features
    scaler = StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)

#  Evaluation
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    lg_model=LogisticRegression(max_iter=1000)
    lg_model.fit(X_train_scaled,Y_train)
    Y_pred=lg_model.predict(X_test_scaled)
    accuracy=accuracy_score(Y_test,Y_pred)
    print("Accuracy: ", round(accuracy,4))
    print("Classification report:\n", classification_report(Y_test,Y_pred))
    print("Confusion Matrix:\n", confusion_matrix(Y_test,Y_pred))

    from sklearn.metrics import ConfusionMatrixDisplay
    cm=confusion_matrix(Y_test,Y_pred)
    disp=ConfusionMatrixDisplay(cm,display_labels=['No Default','Default'])
    disp.plot(cmap='Blues')
    plt.title('Logistic regression -Confusion Matrix')
    plt.show()

#  factors influencing loan approval for each model
    coefficients= pd.DataFrame({'Feature':X.columns,
                                'Coefficient':lg_model.coef_[0],
                                'Impact':np.where(lg_model.coef_[0] >0,'Increases approval likelihoode','Decreases approval likelihoode')})
    
    coefficients=coefficients.reindex(coefficients['Coefficient'].abs().sort_values(ascending=False).index)
    print("\nFeature Importance:")
    print(coefficients)

#  Prediction Examples1
    sample_data= scaler.transform([[35, #age
    500000, # income
    30000, # loan amount
    15.5,  # interest rate
    600,   # credit score
    1      # previous default
    ]])
    prediction=lg_model.predict(sample_data)[0]
    print(f"\n Sample Prediction(High Risk Applicant): {prediction:.1%} probability of default")

#  Prediction Examples2
    sample_data= scaler.transform([[35, #age
    500000, #income
    300000, #loan amount (changed)
    15.5,  #interest rate
    600,   #credit score
    1      #previous default
    ]])                               
    prediction=lg_model.predict(sample_data)[0]
    print(f"\n Sample Prediction(High Risk Applicant): {prediction:.1%} probability of default")


2. Decision tree model

# scaling / Split data into testing and training
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
    dt_model=DecisionTreeClassifier(max_depth=3, random_state=42)
    dt_model.fit(X_train,Y_train)

#  Evaluation
    dt_pred=dt_model.predict(X_test)
    print("Decision Tree Model:")
    print(f"Accuracy:{accuracy_score(Y_test,dt_pred):.2f}")
    print("Confusion Matrix:\n", confusion_matrix(Y_test,dt_pred))
    print("Classification report:\n", classification_report(Y_test,dt_pred))

#  Confusion Matrix
    from sklearn.metrics import ConfusionMatrixDisplay
    cm=confusion_matrix(Y_test,dt_pred)
    disp=ConfusionMatrixDisplay(cm,display_labels=['No Default','Default'])
    disp.plot(cmap='Blues')
    plt.title('Decision tree -Confusion Matrix')
    plt.show()
                

    from sklearn.tree import plot_tree
    plt.figure(figsize=(15,8))
    plot_tree(dt_model,
              feature_names=features,
              class_names=['No Default','Default'],
              filled=True,
              rounded=True)
    plt.title("Decision Tree Visualization")
    plt.show()

#  factors influencing loan approval for each model
    importances=dt_model.feature_importances_
    feature_importance= pd.DataFrame({'Feature':X.columns,
                                'Importance': importances})
    
    feature_importance=feature_importance.sort_values(by='Importance', ascending=False)
    print("\nFeature Importance:")
    print(feature_importance)

#  Prediction Examples1
    sample_data= scaler.transform([[35, #age
    500000, #income
    30000, #loan amount
    15.5,  #interest rate
    600,   #credit score
    1      #previous default
    ]])
    prediction=dt_model.predict(sample_data)[0]
    print(f"\n Sample Prediction(High Risk Applicant): {prediction:} probability of default")

#  Prediction Examples2
    sample_data= scaler.transform([[35, #age
    500000, #income
    30000, #loan amount
    15.5,  #interest rate
    600,   #credit score
    0      #previous default (change the default value)
    ]])
    prediction=dt_model.predict(sample_data)[0]
    print(f"\n Sample Prediction(High Risk Applicant): {prediction:} probability of default")

3. KNN Model
   
#  scaling Features
    scaler=StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)

#  Test and Train model
    from sklearn.neighbors import KNeighborsClassifier
    knn_model =KNeighborsClassifier(n_neighbors=7)
    knn_model.fit(X_train_scaled, Y_train)

#  predictions 
    knn_pred=knn_model.predict(X_test_scaled)

#  Evaluate model
    print("KNN Model:")
    print(f"Accuracy:{accuracy_score(Y_test, knn_pred):.2f}")
    print("Confusion Matrix:\n", confusion_matrix(Y_test,knn_pred))

#  confusion Matrix
    cm=confusion_matrix(Y_test,knn_pred)
    disp=ConfusionMatrixDisplay(cm,display_labels=['No default','Default'])
    disp.plot(cmap='Reds')
    plt.title('KNN - Confusion Matrix')
    plt.show()

#  K value
    k_values = range(1,20)
    accuracies = []
    for k in k_values:
      knn=KNeighborsClassifier(n_neighbors=k)
      knn.fit(X_train_scaled,Y_train)
      pred= knn.predict(X_test_scaled)
      accuracies.append(accuracy_score(Y_test,pred))
    plt.plot(k_values, accuracies, marker='o',color='purple')
    plt.title('KNN Accuracy vs K Value')
    plt.xlabel('Number of Neighbors(K)')
    plt.ylabel('Accuracy')
    plt.xticks(k_values)
    plt.grid()
    plt.show()
            
#  factors influencing loan approval for each model

# feature selection by Permutation_importance
    from sklearn.inspection import permutation_importance
    result=permutation_importance(knn_model,X,Y,n_repeats=10,random_state=42)
    feature_selection= pd.DataFrame({'Feature':X.columns,
                                'Importance': result.importances_mean})
    
    feature_selection=feature_importance.sort_values(by='Importance', ascending=False)
    print("\nFeature Importance:")
    print(feature_importance)

#  Prediction Examples1
    sample_data= scaler.transform([[35, #age
    500000, #income
    30000, #loan amount
    15.5,  #interest rate
    600,   #credit score
    1      #previous default
    ]])
    prediction=knn_model.predict(sample_data)[0]
    print(f"\n Sample Prediction(High Risk Applicant): {prediction:} probability of default")

#  Prediction Examples2
    sample_data= scaler.transform([[35, #age
    500000, #income
    30000, #loan amount
    15.5,  #interest rate
    600,   #credit score
    0      #previous default (changed)
    ]])
    prediction=dt_model.predict(sample_data)[0]
    print(f"\n Sample Prediction(High Risk Applicant): {prediction:} probability of default")



Key Findings

About 22% of loans were approved (loan_status = 1).

Income distribution shows approved loans tend to have higher incomes.


 
                    
    
    
                    
