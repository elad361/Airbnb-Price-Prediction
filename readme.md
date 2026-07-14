# Airbnb Price Prediction -- NYC

**Student:** Elad Shoham | **I.D:** 205439649 | **Email:** elads361@gmail.com

A machine learning project that predicts the nightly price of Airbnb listings
in New York City, built for the *Theoretical Data Science* course.

The project demonstrates core theoretical concepts from the course:
k-fold cross-validation, bias-variance tradeoff, regularisation (SRM),
VC dimension, PAC learning, feature correlation, and residual diagnostics.

## Files

| File | Description |
|------|-------------|
| `airbnb.py` | Main analysis: cleans data, EDA, 5-fold CV, learning curves, residuals, saves artifacts |
| `predict.py` | Interactive price estimator (requires `airbnb.py` to run first) |
| `Airbnb_Report.pdf` | Full project report with all plots and explanations |
| `listings.csv.gz` | Raw listings data from Inside Airbnb (NYC, 2024) |
| `calendar.csv.gz` | Raw calendar data from Inside Airbnb (NYC, 2024) |

## How to Run

### 1. Install dependencies (once)
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy contextily pyproj fpdf2
```

### 2. Run the main analysis
```bash
python airbnb.py
```
Produces plots, `results.json`, `model.pkl`, and `encoders.pkl`.
Takes a few minutes (learning curves train multiple models at different dataset sizes).

### 3. Predict a price for a specific apartment
```bash
python predict.py
```
Loads `model.pkl` and `encoders.pkl` -- no re-training needed.

## Data

- **Source:** [Inside Airbnb](http://insideairbnb.com/get-the-data/) -- New York City, 2024
- **Raw size:** 35,036 listings, 90 columns
- **After cleaning:** 19,859 listings

## Theoretical Concepts Applied

| Concept | Where in the project |
|---------|----------------------|
| Bias-variance tradeoff | Learning curves (PAC: approximation vs estimation error) |
| K-fold cross-validation | 5-fold CV on training set; mean justified by CLT |
| No Free Lunch theorem | Reason for comparing three different models |
| Structural Risk Minimisation | Ridge regression: training error + alpha * ||w||^2 |
| VC dimension | Linear models: VC dim = d+1 = 13; explains high bias |
| Feature scale normalisation | StandardScaler pipeline for linear models |
| Multicollinearity | Correlation matrix + Pearson r with p-values (scipy) |
| Residual diagnostics | Shapiro-Wilk normality test + residuals vs fitted plot |
| Imputation | Median imputation for bedrooms, beds, bathrooms, review score |

## Model Comparison (5-Fold CV, log-scale MAE)

| Model | CV MAE | CV R2 |
|-------|--------|-------|
| Linear Regression | 0.411 +/- 0.005 | 0.520 +/- 0.021 |
| Ridge Regression | 0.411 +/- 0.005 | 0.520 +/- 0.021 |
| **Gradient Boosting** | **0.319 +/- 0.005** | **0.704 +/- 0.011** |

Error bars show standard deviation across the 5 folds.

## Bias-Variance Summary

| Model | Diagnosis | Explanation |
|-------|-----------|-------------|
| Linear Regression | High bias (underfitting) | VC dim = 13, too simple to capture non-linear pricing |
| Ridge Regression | High bias (underfitting) | Same hypothesis class as Linear Regression |
| Gradient Boosting | Balanced | Lower bias, controlled variance via max_depth and learning_rate |

## Key Findings

- **Location** (latitude, longitude) is the strongest predictor -- exact position matters more than neighbourhood name
- **Apartment size** (bedrooms, accommodates) is the second strongest group of predictors
- **Review score** has very weak predictive power -- NYC supply constraints let hosts charge regardless of rating
- **Residual analysis** shows heteroscedasticity at high prices -- luxury listings are harder to predict
- **Shapiro-Wilk test** confirms residuals are not perfectly normal (p << 0.05), due to the heavy right tail

## Final Result

- Test MAE: $72 per night
- Test R2: 0.570
