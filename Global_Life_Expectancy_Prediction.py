# ==============================================================================
# GLOBAL LIFE EXPECTANCY PREDICTION
# ==============================================================================

# ==============================================================================
# IMPORT LIBRARIES
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


# ==============================================================================
# LOAD DATASET
# ==============================================================================

df = pd.read_csv('data/Life Expectancy Data.csv')

print("\nFirst 5 rows:")
print(df.head())


# ==============================================================================
# BASIC DATA EXPLORATION
# ==============================================================================

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Columns:")
print(df.columns)

print("\nDataset Information:")
print(df.info())

print("\nStatistical Description:")
print(df.describe())


# ==============================================================================
# DATA CLEANING
# ==============================================================================

# Fixing feature names
df.rename(columns=lambda x: x.replace(' ', ''), inplace=True)

print("\nColumns after removing spaces:")
print(df.columns)

# Fix incorrect column name
df.rename(
    columns={'thinness1-19years': 'thinness10-19years'},
    inplace=True
)

print("\nColumns after correcting thinness feature:")
print(df.columns)


# ==============================================================================
# CHECK DUPLICATES
# ==============================================================================

print("\nNumber of duplicate rows:")
print(df.duplicated().sum())


# ==============================================================================
# CHECK MISSING VALUES
# ==============================================================================

print("\nMissing values:")
print(df.isnull().sum())

missing_values_percentage = df.isnull().sum() / len(df) * 100

print("\nMissing values percentage:")
print(missing_values_percentage)


# ==============================================================================
# UNIQUE VALUES
# ==============================================================================

print("\nNumber of unique values:")
print(df.nunique())


# ==============================================================================
# EXPLORATORY DATA ANALYSIS
# ==============================================================================

# Histogram and boxplot for Life Expectancy

plt.figure(figsize=(12, 3))

plt.subplot(1, 2, 1)
sns.histplot(
    data=df,
    x='Lifeexpectancy',
    kde=True,
    edgecolor='red'
)
plt.title("Distribution of Life Expectancy")

plt.subplot(1, 2, 2)
sns.boxplot(
    data=df,
    x='Lifeexpectancy'
)
plt.title("Boxplot of Life Expectancy")

plt.tight_layout()
plt.show()


# ==============================================================================
# COUNTRY AND STATUS DISTRIBUTION
# ==============================================================================

print("\nCountry value counts:")
print(df['Country'].value_counts())

print("\nStatus value counts:")
print(df['Status'].value_counts())

print("\nColumns:")
print(df.columns)


# ==============================================================================
# HISTOGRAM AND BOXPLOT FOR ALL NUMERICAL COLUMNS
# ==============================================================================

for i in df.columns:

    if df[i].dtype == 'float64' or df[i].dtype == 'int64':

        plt.figure(figsize=(12, 3))

        # Histogram
        plt.subplot(1, 2, 1)

        sns.histplot(
            data=df,
            x=i,
            kde=True,
            edgecolor='red'
        )

        plt.title(f"Histogram of {i}")

        # Boxplot
        plt.subplot(1, 2, 2)

        sns.boxplot(
            data=df,
            x=i
        )

        plt.title(f"Boxplot of {i}")

        plt.tight_layout()
        plt.show()


# ==============================================================================
# LIFE EXPECTANCY VS STATUS
# ==============================================================================

plt.figure(figsize=(12, 3))

sns.barplot(
    data=df,
    x='Status',
    y='Lifeexpectancy'
)

plt.title("Life Expectancy vs Status")
plt.tight_layout()
plt.show()


# ==============================================================================
# SCATTER PLOTS - NUMERICAL VARIABLES VS LIFE EXPECTANCY
# ==============================================================================

numeric_columns = [
    'AdultMortality',
    'infantdeaths',
    'Alcohol',
    'percentageexpenditure',
    'HepatitisB',
    'Measles',
    'BMI',
    'under-fivedeaths',
    'Polio',
    'Totalexpenditure',
    'Diphtheria',
    'HIV/AIDS',
    'GDP',
    'Population',
    'thinness10-19years',
    'thinness5-9years',
    'Incomecompositionofresources',
    'Schooling'
]

plt.figure(figsize=(15, 10))

for i, col in enumerate(numeric_columns, 1):

    plt.subplot(4, 5, i)

    sns.scatterplot(
        x=df[col],
        y=df['Lifeexpectancy']
    )

    plt.title(f"Lifeexpectancy vs {col}")
    plt.xlabel(col)
    plt.ylabel('Lifeexpectancy')

plt.tight_layout()
plt.show()


# ==============================================================================
# CHECK DATA BEFORE PREPROCESSING
# ==============================================================================

print("\nDataset Shape:")
print(df.shape)

print("\nTotal Missing Values:")
print(df.isnull().sum().sum())

print("\nColumns:")
print(df.columns)


# ==============================================================================
# MISSING VALUE IMPUTATION
# ==============================================================================

from sklearn.impute import SimpleImputer

# Columns where mean imputation is used
mean_median_cols = [
    'Alcohol',
    'BMI',
    'Totalexpenditure',
    'thinness10-19years',
    'thinness5-9years'
]

# Columns where mode imputation is used
mode_cols = [
    'HepatitisB',
    'Polio',
    'Diphtheria'
]


# Mean and median imputers
mean_imputer = SimpleImputer(strategy='mean')
median_imputer = SimpleImputer(strategy='median')


# Mean imputation
df[mean_median_cols] = mean_imputer.fit_transform(
    df[mean_median_cols]
)


# Mode imputation
mode_imputer = SimpleImputer(
    strategy='most_frequent'
)

df[mode_cols] = mode_imputer.fit_transform(
    df[mode_cols]
)


# ==============================================================================
# CHECK MISSING VALUES AFTER IMPUTATION
# ==============================================================================

print("\nTotal missing values after initial imputation:")
print(df.isnull().sum().sum())

print("\nMissing values by column:")
print(df.isnull().sum())


# ==============================================================================
# CHECK ROWS WHERE LIFE EXPECTANCY IS MISSING
# ==============================================================================

print("\nRows with missing Life Expectancy:")
print(df[df['Lifeexpectancy'].isna()])


# ==============================================================================
# DROP ROWS WITH MISSING TARGET VALUES
# ==============================================================================

df.dropna(
    subset=['Lifeexpectancy'],
    inplace=True
)

print("\nMissing values after removing missing Life Expectancy:")
print(df.isna().sum())


# ==============================================================================
# INTERPOLATION FOR GDP AND POPULATION
# ==============================================================================

df['GDP'].interpolate(
    method='linear',
    inplace=True
)

df['Population'].interpolate(
    method='linear',
    inplace=True
)

print("\nMissing values after GDP and Population interpolation:")
print(df.isna().sum())


# ==============================================================================
# FILL SCHOOLING AND INCOME COMPOSITION VALUES
# ==============================================================================

df['Incomecompositionofresources'].fillna(
    df.groupby('Status')['Incomecompositionofresources']
    .transform('median'),
    inplace=True
)

df['Schooling'].fillna(
    df.groupby('Status')['Schooling']
    .transform('median'),
    inplace=True
)

print("\nMissing values after final missing-value treatment:")
print(df.isna().sum())

print("\nDataset shape:")
print(df.shape)


# ==============================================================================
# UNIQUE COUNTRIES
# ==============================================================================

print("\nUnique Countries:")
print(df['Country'].unique())


# ==============================================================================
# HISTOGRAM FOR NUMERICAL FEATURES AFTER IMPUTATION
# ==============================================================================

for i in df.columns:

    if df[i].dtype == 'float64' or df[i].dtype == 'int64':

        plt.figure(figsize=(12, 3))

        plt.subplot(1, 2, 1)

        sns.histplot(
            data=df,
            x=i,
            kde=True,
            edgecolor='red'
        )

        plt.title(f"Histogram of {i}")

        plt.subplot(1, 2, 2)

        sns.boxplot(
            data=df,
            x=i
        )

        plt.title(f"Boxplot of {i}")

        plt.tight_layout()
        plt.show()


# ==============================================================================
# BOXPLOTS FOR NUMERICAL FEATURES
# ==============================================================================

for i in df.columns:

    if df[i].dtype == 'float64' or df[i].dtype == 'int64':

        plt.figure(figsize=(8, 3))

        sns.boxplot(
            data=df,
            x=i
        )

        plt.title(f"Boxplot of {i}")

        plt.tight_layout()
        plt.show()


# ==============================================================================
# OUTLIER TREATMENT
# ==============================================================================

cols_to_handle_outliers = [
    'AdultMortality',
    'infantdeaths',
    'Alcohol',
    'percentageexpenditure',
    'HepatitisB',
    'Measles',
    'BMI',
    'under-fivedeaths',
    'Polio',
    'Totalexpenditure',
    'Diphtheria',
    'HIV/AIDS',
    'GDP',
    'Population',
    'thinness10-19years',
    'thinness5-9years',
    'Incomecompositionofresources',
    'Schooling'
]

for col_name in cols_to_handle_outliers:

    # Calculate quartiles
    q1 = df[col_name].quantile(0.25)
    q3 = df[col_name].quantile(0.75)

    # Calculate IQR
    iqr = q3 - q1

    # Calculate bounds
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Replace outliers with column mean
    df[col_name] = np.where(
        (df[col_name] > upper_bound) |
        (df[col_name] < lower_bound),
        np.mean(df[col_name]),
        df[col_name]
    )


# ==============================================================================
# BOXPLOTS AFTER OUTLIER TREATMENT
# ==============================================================================

for i in df.columns:

    if df[i].dtype == 'float64' or df[i].dtype == 'int64':

        plt.figure(figsize=(8, 3))

        sns.boxplot(
            data=df,
            x=i
        )

        plt.title(f"Boxplot after Outlier Treatment - {i}")

        plt.tight_layout()
        plt.show()


# ==============================================================================
# DATASET INFORMATION AFTER PREPROCESSING
# ==============================================================================

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nUnique values:")
print(df.nunique())

print("\nStatus distribution:")
print(df['Status'].value_counts())


# ==============================================================================
# BINARY ENCODING OF STATUS
# ==============================================================================

df['Status'] = df['Status'].map({
    'Developed': 1,
    'Developing': 0
})

df['Status'] = df['Status'].astype(int)

print("\nData after Status encoding:")
print(df.head())


# ==============================================================================
# TARGET ENCODING OF COUNTRY
# ==============================================================================

# Calculate mean Life Expectancy for each country
category_means = (
    df.groupby('Country')['Lifeexpectancy']
    .mean()
    .round(2)
    .to_dict()
)

# Map country mean Life Expectancy back to Country
df['Country'] = df['Country'].map(category_means)

print("\nData after Country target encoding:")
print(df.head())

print("\nDataset information:")
print(df.info())


# ==============================================================================
# CORRELATION HEATMAP
# ==============================================================================

plt.figure(figsize=(15, 6))

sns.heatmap(
    df.corr(),
    annot=True
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# ==============================================================================
# DROP YEAR
# ==============================================================================

df.drop(
    ['Year'],
    axis=1,
    inplace=True
)

print("\nDataset shape after dropping Year:")
print(df.shape)


# ==============================================================================
# DEFINE FEATURES AND TARGET
# ==============================================================================

X = df.drop(
    columns=["Lifeexpectancy"]
)

y = df["Lifeexpectancy"]

print("\nFeatures:")
print(X)

print("\nTarget:")
print(y)


# ==============================================================================
# TRAIN TEST SPLIT
# ==============================================================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining and testing shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


# ==============================================================================
# FEATURE SCALING
# ==============================================================================

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=X_train.columns
)

print("\nScaled training data:")
print(X_train_scaled.head())


# ==============================================================================
# LINEAR REGRESSION
# ==============================================================================

from sklearn.linear_model import LinearRegression

modelLR = LinearRegression()

modelLR.fit(
    X_train_scaled,
    y_train
)

y_pred = modelLR.predict(
    X_test_scaled
)


# ==============================================================================
# MODEL EVALUATION FUNCTION
# ==============================================================================

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def evaluate_model(y_test, y_pred):

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        y_pred
    )

    scores = {
        "Mean Squared Error": mse,
        "Mean Absolute Error": mae,
        "Root Mean Squared Error": rmse,
        "R2 Score": r2
    }

    return scores


print("\nLinear Regression Results:")
print(evaluate_model(y_test, y_pred))


# ==============================================================================
# KNN REGRESSION
# ==============================================================================

from sklearn.neighbors import KNeighborsRegressor

modelKNN = KNeighborsRegressor(
    n_neighbors=5,
    n_jobs=-1
)

modelKNN.fit(
    X_train_scaled,
    y_train
)

y_pred_knn = modelKNN.predict(
    X_test_scaled
)

print("\nKNN Regression Results:")
print(evaluate_model(y_test, y_pred_knn))


# ==============================================================================
# FIND OPTIMAL K VALUE
# ==============================================================================

train_mse = []
test_mse = []

k_values = range(1, 20)

for k in k_values:

    knn = KNeighborsRegressor(
        n_neighbors=k,
        n_jobs=-1
    )

    knn.fit(
        X_train_scaled,
        y_train
    )

    # Training predictions
    y_train_pred = knn.predict(
        X_train_scaled
    )

    # Testing predictions
    y_test_pred = knn.predict(
        X_test_scaled
    )

    # Training MSE
    train_mse.append(
        mean_squared_error(
            y_train,
            y_train_pred
        )
    )

    # Testing MSE
    test_mse.append(
        mean_squared_error(
            y_test,
            y_test_pred
        )
    )


plt.figure(figsize=(10, 4))

plt.plot(
    k_values,
    train_mse,
    label='Training MSE',
    marker='o'
)

plt.plot(
    k_values,
    test_mse,
    label='Testing MSE',
    marker='o'
)

plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Mean Squared Error')
plt.title('MSE for Training and Testing Data vs. k')

plt.legend()
plt.grid()

plt.show()


# ==============================================================================
# SUPPORT VECTOR REGRESSION
# ==============================================================================

from sklearn.svm import SVR

modelSVR = SVR()

modelSVR.fit(
    X_train_scaled,
    y_train
)

y_pred_SVR = modelSVR.predict(
    X_test_scaled
)

print("\nSVR Results:")
print(evaluate_model(y_test, y_pred_SVR))


# ==============================================================================
# DECISION TREE REGRESSION
# ==============================================================================

from sklearn.tree import DecisionTreeRegressor

modelDTR = DecisionTreeRegressor(
    random_state=42
)

modelDTR.fit(
    X_train,
    y_train
)

y_pred_DTR = modelDTR.predict(
    X_test
)

print("\nDecision Tree Results:")
print(evaluate_model(y_test, y_pred_DTR))


# ==============================================================================
# LASSO REGRESSION
# ==============================================================================

from sklearn.linear_model import Lasso

lasso = Lasso(
    alpha=0.001
)

lasso.fit(
    X_train_scaled,
    y_train
)

print("\nLasso Coefficients:")

for feature, coef in zip(
    X.columns,
    lasso.coef_
):

    print(
        f"{feature}: {coef:.4f}"
    )


# ==============================================================================
# FEATURE SELECTION
# ==============================================================================

print("\nCurrent Dataset Columns:")
print(df.columns)

print("\nCurrent X Columns:")
print(X.columns)


# Drop Country feature
selected_features = X.drop(
    columns=['Country']
)

print("\nSelected Features:")
print(selected_features.head())


# Update X and y
X = selected_features

y = df["Lifeexpectancy"]

print("\nUpdated Features:")
print(X)


# ==============================================================================
# TRAIN TEST SPLIT AFTER FEATURE SELECTION
# ==============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nShapes after feature selection:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


# ==============================================================================
# DATA SCALING AFTER FEATURE SELECTION
# ==============================================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ==============================================================================
# LINEAR REGRESSION AFTER FEATURE SELECTION
# ==============================================================================

lasso_modelLR = LinearRegression()

lasso_modelLR.fit(
    X_train_scaled,
    y_train
)

y_pred = lasso_modelLR.predict(
    X_test_scaled
)

print("\nLinear Regression after feature selection:")
print(
    evaluate_model(
        y_test,
        y_pred
    )
)


results = evaluate_model(
    y_test,
    y_pred
)

R2_LR = results['R2 Score']

print("\nLinear Regression R2 Score:")
print(R2_LR)


# ==============================================================================
# KNN AFTER FEATURE SELECTION
# ==============================================================================

lasso_modelKNN = KNeighborsRegressor(
    n_neighbors=5,
    n_jobs=-1
)

lasso_modelKNN.fit(
    X_train_scaled,
    y_train
)

y_pred_knn = lasso_modelKNN.predict(
    X_test_scaled
)

print("\nKNN after feature selection:")
print(
    evaluate_model(
        y_test,
        y_pred_knn
    )
)


# ==============================================================================
# SVR AFTER FEATURE SELECTION
# ==============================================================================

lasso_modelSVR = SVR()

lasso_modelSVR.fit(
    X_train_scaled,
    y_train
)

y_pred_SVR = lasso_modelSVR.predict(
    X_test_scaled
)

print("\nSVR after feature selection:")
print(
    evaluate_model(
        y_test,
        y_pred_SVR
    )
)


# ==============================================================================
# DECISION TREE AFTER FEATURE SELECTION
# ==============================================================================

lasso_modelDTR = DecisionTreeRegressor(
    random_state=42
)

lasso_modelDTR.fit(
    X_train,
    y_train
)

y_pred_DTR = lasso_modelDTR.predict(
    X_test
)

print("\nDecision Tree after feature selection:")
print(
    evaluate_model(
        y_test,
        y_pred_DTR
    )
)


# ==============================================================================
# GRID SEARCH CROSS VALIDATION
# ==============================================================================

from sklearn.model_selection import GridSearchCV


# ==============================================================================
# DECISION TREE HYPERPARAMETER TUNING
# ==============================================================================

param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2']
}


dt = DecisionTreeRegressor(
    random_state=42
)


grid_search = GridSearchCV(
    estimator=dt,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    verbose=1,
    n_jobs=-1
)


grid_search.fit(
    X_train,
    y_train
)


# Best parameters
print("\nBest Decision Tree Parameters:")
print(grid_search.best_params_)


# Best model
best_dt = grid_search.best_estimator_


# Predictions
y_pred_GS_DTR = best_dt.predict(
    X_test
)


# Evaluation
print("\nOptimized Decision Tree Results:")
print(
    evaluate_model(
        y_test,
        y_pred_GS_DTR
    )
)


results = evaluate_model(
    y_test,
    y_pred_GS_DTR
)

R2_DTR = results['R2 Score']

print("\nOptimized Decision Tree R2 Score:")
print(R2_DTR)


# ==============================================================================
# KNN HYPERPARAMETER TUNING
# ==============================================================================

knn_params = {
    'n_neighbors': [5, 7, 10, 12, 15],
    'weights': ['uniform', 'distance'],
    'metric': [
        'euclidean',
        'manhattan',
        'minkowski'
    ]
}


knn = KNeighborsRegressor()


grid_knn = GridSearchCV(
    knn,
    knn_params,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)


grid_knn.fit(
    X_train_scaled,
    y_train
)


# Best parameters
print("\nBest KNN Parameters:")
print(grid_knn.best_params_)


# Best model
best_knn = grid_knn.best_estimator_


# Prediction
y_pred_GS_KNN = best_knn.predict(
    X_test_scaled
)


# Evaluation
print("\nOptimized KNN Results:")
print(
    evaluate_model(
        y_test,
        y_pred_GS_KNN
    )
)


results = evaluate_model(
    y_test,
    y_pred_GS_KNN
)

R2_KNN = results['R2 Score']

print("\nOptimized KNN R2 Score:")
print(R2_KNN)


# ==============================================================================
# SVR HYPERPARAMETER TUNING
# ==============================================================================

svr_params = {
    'kernel': [
        'linear',
        'rbf'
    ],
    'C': [
        0.1,
        1,
        10
    ],
    'epsilon': [
        0.1,
        0.2,
        0.5,
        1.0
    ]
}


svr = SVR()


grid_svr = GridSearchCV(
    svr,
    svr_params,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)


grid_svr.fit(
    X_train_scaled,
    y_train
)


# Best model
print("\nBest SVR Model:")
print(grid_svr.best_estimator_)


best_svr = grid_svr.best_estimator_


# Prediction
y_pred_GS_SVR = best_svr.predict(
    X_test_scaled
)


# Evaluation
print("\nOptimized SVR Results:")
print(
    evaluate_model(
        y_test,
        y_pred_GS_SVR
    )
)


results = evaluate_model(
    y_test,
    y_pred_GS_SVR
)

R2_SVR = results['R2 Score']

print("\nOptimized SVR R2 Score:")
print(R2_SVR)


# ==============================================================================
# FINAL MODEL COMPARISON
# ==============================================================================

model_names = [
    'Linear Regression',
    'KNN',
    'SVR',
    'DTR'
]

model_scores = [
    R2_LR,
    R2_KNN,
    R2_SVR,
    R2_DTR
]


print("\nFinal Model R2 Scores:")

for model_name, score in zip(
    model_names,
    model_scores
):

    print(
        f"{model_name}: {score:.4f}"
    )


# ==============================================================================
# MODEL COMPARISON BAR GRAPH
# ==============================================================================

plt.figure(figsize=(10, 5))

plt.bar(
    model_names,
    model_scores,
    color='skyblue'
)

plt.xlabel('Model Names')
plt.ylabel('R2 Scores')
plt.title('Comparison of Model Scores')

plt.tight_layout()
plt.show()


# ==============================================================================
# END OF PROJECT
# ==============================================================================