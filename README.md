# House Price Prediction – Melbourne (Regression Models & Random Forest)

This project predicts house prices using the Melbourne Housing Snapshot dataset and compares several models, including a Random Forest Regressor.

## Project Overview

The objective is to build a machine learning pipeline that:

- Cleans and preprocesses the raw housing data  
- Trains multiple regression models to predict the sale price  
- Compares their performance using R² and RMSE  
- Uses the best model (Random Forest) to predict the price of a hypothetical house  

Everything is implemented in a single Jupyter notebook: **`House-Price-Prediction.ipynb`**.

## Dataset

- File: `melb_data.csv`  
- Number of rows: 13,580  
- Original columns (21 total) include:  
  `Suburb`, `Address`, `Rooms`, `Type`, `Price`, `Method`, `SellerG`, `Date`, `Distance`, `Postcode`, `Bedroom2`, `Bathroom`, `Car`, `Landsize`, `BuildingArea`, `YearBuilt`, `CouncilArea`, `Lattitude`, `Longtitude`, `Regionname`, `Propertycount`  

- Target variable:  
  - `Price` – numeric sale price  

## Workflow

### 1. Imports and Data Loading

The notebook imports standard Python data science libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`) and scikit‑learn tools for preprocessing, model training, and evaluation.

It then loads `melb_data.csv` and prints the shape, first rows, and detailed info (column names, types, non‑null counts). This confirms 13,580 rows and 21 columns, with several columns containing missing values.

### 2. Missing Values and Column Selection

Missing values per column are summarized using a table of counts.

To simplify the model and remove high‑missing or less informative fields, the following columns are dropped:

- `Suburb`  
- `SellerG`  
- `Postcode`  
- `Address`  
- `Date`  
- `CouncilArea`  
- `BuildingArea`  
- `YearBuilt`  

The remaining columns are:

- `Rooms`, `Type`, `Price`, `Method`, `Distance`, `Bedroom2`, `Bathroom`, `Car`, `Landsize`, `Lattitude`, `Longtitude`, `Regionname`, `Propertycount`  

Missing values in `Car` are then filled using the most frequent value in that column. Any remaining rows with missing values are removed so that the modeling steps receive a clean dataset.

### 3. Encoding and Train/Test Split

Categorical variables are encoded to numeric form using label encoding for:

- `Type`  
- `Method`  
- `Regionname`  

The feature matrix and target vector are then defined as:

- Features (`X`): all columns except `Price`  
- Target (`y`): `Price`  

The data is split into training and test sets using an 80/20 split with a fixed random state to ensure reproducibility.

### 4. Feature Scaling

For the linear models, the feature values are standardized using `StandardScaler`:

- The scaler is fit on the training data  
- Both training and test feature sets are transformed  

This improves the stability of Linear, Ridge, and Lasso regression. The Random Forest model is trained on the original (unscaled) features, because tree‑based models do not require scaling.

### 5. Models Trained

The notebook trains and evaluates **four regression models**:
  
1. Ridge Regression  
2. Lasso Regression  
3. Random Forest Regressor  

For each model:

- The model is fit on the training data  
- Predictions are generated on the test set  
- Two evaluation metrics are computed:
  - **R²** (coefficient of determination)  
  - **RMSE** (root mean squared error)  

Example results:
 
- Ridge Regression:  R² ≈ 0.52, RMSE ≈ 437,000  
- Lasso Regression:  R² ≈ 0.52, RMSE ≈ 437,000  
- Random Forest: **R² ≈ 0.82, RMSE ≈ 265,000**  

The three linear models perform similarly and explain about half of the variance in house prices, while the Random Forest explains over 80% and significantly reduces prediction error. Therefore, Random Forest is selected as the **final model**.

### 6. Visual Evaluation

To visually assess model performance, the notebook includes **scatter plots of actual vs predicted prices**:

- A plot for Ridge Regression  
- A plot for Random Forest  

Each plot:

- Uses actual prices on one axis and predicted prices on the other  
- Includes a diagonal reference line representing perfect predictions  

The Random Forest scatter plot shows points much closer to this diagonal compared to the Ridge plot, visually confirming the superior performance of the Random Forest model.

### 7. Predicting a Hypothetical House

The final part of the notebook demonstrates how to use the trained Random Forest model to predict the price of a **hypothetical house**.

Steps shown:

- Define an example property with specified values for:
  - Number of rooms, property type (`h`), sale method (`S`), distance to CBD, number of bedrooms and bathrooms, car spaces, land size, latitude, longitude, region name (`Northern Metropolitan`), and property count  
- Convert the example into a one‑row DataFrame  
- Apply the same label encoders used during training so that categorical values are transformed consistently  
- Ensure columns are in the same order as the training features  
- Use the trained Random Forest model to predict the price  

In the notebook run, the example house receives a predicted price of approximately **1,181,282** (in the dataset’s currency units).

## How to Run

1. Place `House-Price-Prediction.ipynb` and `melb_data.csv` in the same folder (or update the path inside the notebook).  
2. Install required Python packages (e.g. via `pip` or conda):
   - `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`  
3. Open the notebook in Jupyter (or VS Code, JupyterLab, etc.).  
4. Run all cells in order:
   - Data loading and exploration  
   - Missing value analysis and column dropping  
   - Encoding and splitting  
   - Scaling and model training  
   - Metric comparison and visualizations  
   - Hypothetical example prediction  

## Summary

- The project uses a cleaned subset of the Melbourne Housing data to predict sale prices  
- It compares classical linear models (Linear, Ridge, Lasso) with a Random Forest Regressor  
- Random Forest substantially outperforms the linear models and is chosen as the final model  
- The notebook includes a concrete example that predicts the price of a specific hypothetical property using the trained Random Forest  
