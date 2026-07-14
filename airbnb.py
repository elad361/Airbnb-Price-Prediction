import json
import pickle

import contextily as ctx
from pyproj import Transformer
from scipy import stats

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import (GridSearchCV, KFold, cross_val_score,
                                     learning_curve, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── 1. LOAD & CLEAN ───────────────────────────────────────────────────────────
df = pd.read_csv("listings.csv.gz", compression='gzip', low_memory=False)
df['price'] = df['price'].replace(r'[\$,]', '', regex=True).astype(float)
df = df[(df['price'] > 0) & (df['price'] < 2000)]

cols = ['price', 'neighbourhood_cleansed', 'room_type', 'accommodates',
        'bedrooms', 'beds', 'bathrooms', 'review_scores_rating',
        'number_of_reviews', 'minimum_nights', 'latitude', 'longitude',
        'host_is_superhost']
df = df[cols].dropna(subset=['price', 'neighbourhood_cleansed', 'room_type',
                              'accommodates', 'number_of_reviews', 'minimum_nights',
                              'latitude', 'longitude', 'host_is_superhost'])

# Impute high-NaN numeric columns with median
for col in ('bedrooms', 'beds', 'bathrooms', 'review_scores_rating'):
    df[col] = df[col].fillna(df[col].median())

assert df.isnull().sum().sum() == 0, "NaN values still present after imputation"

# Boolean column stored as 't'/'f'
df['host_is_superhost'] = (df['host_is_superhost'] == 't').astype(int)

print(f"Listings after cleaning: {len(df):,}")

# ── 2. EDA PLOTS ──────────────────────────────────────────────────────────────
# Plot 1: Price distribution
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df['price'], bins=80, color='steelblue', edgecolor='white', linewidth=0.5)
ax.axvline(df['price'].median(), color='red',    linestyle='--',
           label=f"Median: ${df['price'].median():.0f}")
ax.axvline(df['price'].mean(),   color='orange', linestyle='--',
           label=f"Mean:   ${df['price'].mean():.0f}")
ax.set_xlabel('Price per Night ($)')
ax.set_ylabel('Count')
ax.set_title('Price Distribution (NYC Airbnb 2024)')
ax.legend()
plt.tight_layout()
plt.savefig('plot1_price_dist.png', dpi=150)
plt.close()

# Plot 1b: Log-price distribution
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(np.log1p(df['price']), bins=80, color='steelblue', edgecolor='white', linewidth=0.5)
ax.set_xlabel('log(1 + Price)')
ax.set_ylabel('Count')
ax.set_title('Log-Price Distribution -- much more symmetric')
plt.tight_layout()
plt.savefig('plot1b_logprice_dist.png', dpi=150)
plt.close()

# Plot 2: Room type
room_avg = df.groupby('room_type')['price'].median().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(room_avg.index, room_avg.values, color='steelblue')
ax.set_xlabel('Room Type')
ax.set_ylabel('Median Price per Night ($)')
ax.set_title('Median Price by Room Type')
plt.tight_layout()
plt.savefig('plot2_room_type.png', dpi=150)
plt.close()

# Plot 3: Top 15 neighbourhoods
top15 = df.groupby('neighbourhood_cleansed')['price'].median().nlargest(15)
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(top15.index[::-1], top15.values[::-1], color='steelblue')
ax.set_xlabel('Median Price per Night ($)')
ax.set_title('Top 15 Neighbourhoods by Median Price')
plt.tight_layout()
plt.savefig('plot3_neighbourhood.png', dpi=150)
plt.close()

# Plot 3b: Geographic scatter on real map
# contextily works in Web Mercator (EPSG:3857), so we convert lat/lon first.
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
x, y = transformer.transform(df['longitude'].values, df['latitude'].values)

fig, ax = plt.subplots(figsize=(9, 10))
sc = ax.scatter(x, y, c=np.log1p(df['price']),
                cmap='YlOrRd', alpha=0.4, s=3, linewidths=0)
plt.colorbar(sc, ax=ax, label='log(1 + Price per night)')
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=11)
ax.set_axis_off()
ax.set_title('NYC Airbnb Listings -- colour = price', fontsize=13, pad=10)
plt.tight_layout()
plt.savefig('plot3b_geo.png', dpi=150)
plt.close()
print("EDA plots saved.")

# ── 3. ENCODE CATEGORIES ──────────────────────────────────────────────────────
le_neighbourhood = LabelEncoder()
le_room          = LabelEncoder()
df['neighbourhood_enc'] = le_neighbourhood.fit_transform(df['neighbourhood_cleansed'])
df['room_type_enc']     = le_room.fit_transform(df['room_type'])

# Save neighbourhood -> (lat, lon) medians for predict.py
neighbourhood_coords = (df.groupby('neighbourhood_cleansed')[['latitude', 'longitude']]
                          .median().to_dict(orient='index'))

with open('encoders.pkl', 'wb') as f:
    pickle.dump({
        'neighbourhood':       le_neighbourhood,
        'room_type':           le_room,
        'neighbourhood_coords': neighbourhood_coords,
        'bathrooms_median':    float(df['bathrooms'].median()),
    }, f)

features = ['neighbourhood_enc', 'room_type_enc', 'accommodates',
            'bedrooms', 'beds', 'bathrooms', 'review_scores_rating',
            'number_of_reviews', 'minimum_nights',
            'latitude', 'longitude', 'host_is_superhost']
feature_labels = ['Neighbourhood', 'Room Type', 'Accommodates', 'Bedrooms',
                  'Beds', 'Bathrooms', 'Review Score', 'Num Reviews',
                  'Min Nights', 'Latitude', 'Longitude', 'Superhost']

X = df[features].values
y = df['price'].values

# ── 4. FEATURE CORRELATION ────────────────────────────────────────────────────
numeric_cols = ['price', 'accommodates', 'bedrooms', 'beds', 'bathrooms',
                'review_scores_rating', 'number_of_reviews', 'minimum_nights']
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            mask=mask, ax=ax, square=True, linewidths=0.5)
ax.set_title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('plot6_correlation.png', dpi=150)
plt.close()
print("Correlation plot saved.")

# Pearson r + p-value: each feature vs price (scipy.stats.pearsonr)
pearson_results = {}
price_vals = df['price'].values
for feat, label in zip(features, feature_labels):
    col_vals = df[feat].values if feat in df.columns else df[feat].values
    r, p = stats.pearsonr(col_vals, price_vals)
    pearson_results[label] = {'r': round(float(r), 4), 'p': round(float(p), 6)}

# Plot: Pearson r vs price with significance markers
labels_sorted = sorted(pearson_results, key=lambda k: abs(pearson_results[k]['r']), reverse=True)
r_vals = [pearson_results[k]['r'] for k in labels_sorted]
p_vals = [pearson_results[k]['p'] for k in labels_sorted]
colors_p = ['#2196F3' if p < 0.05 else '#BDBDBD' for p in p_vals]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(labels_sorted[::-1], r_vals[::-1], color=colors_p[::-1])
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Pearson r (correlation with price)')
ax.set_title('Pearson Correlation with Price\n(blue = statistically significant, p < 0.05)')
for bar, p in zip(bars, p_vals[::-1]):
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            sig, va='center', fontsize=9)
plt.tight_layout()
plt.savefig('plot6b_pearson.png', dpi=150)
plt.close()
print("Pearson correlation plot saved.")

# ── 5. LOG-TRANSFORM TARGET ───────────────────────────────────────────────────
# Airbnb prices are right-skewed. Training on log(1+price) makes the target
# more symmetric, reduces the influence of outliers, and fixes heteroscedasticity.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

y_train_log = np.log1p(y_train)

# ── 6. MODELS + 5-FOLD CV ─────────────────────────────────────────────────────
kf = KFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'Linear Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('model',  LinearRegression()),
    ]),
    # Ridge = Tikhonov regularisation (Unit 9). alpha controls regularisation strength.
    'Ridge Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('model',  Ridge(alpha=1.0)),
    ]),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
}

print("\n5-Fold Cross-Validation (log-scale MAE, training set only):")
print(f"{'Model':<22} {'MAE mean':>10} {'MAE std':>9} {'R2 mean':>9} {'R2 std':>8}")
print("-" * 62)

cv_results = {}
for name, model in models.items():
    mae_scores = -cross_val_score(model, X_train, y_train_log, cv=kf,
                                  scoring='neg_mean_absolute_error', n_jobs=-1)
    r2_scores  =  cross_val_score(model, X_train, y_train_log, cv=kf,
                                  scoring='r2', n_jobs=-1)
    cv_results[name] = {
        'mae_mean': float(mae_scores.mean()),
        'mae_std':  float(mae_scores.std()),
        'r2_mean':  float(r2_scores.mean()),
        'r2_std':   float(r2_scores.std()),
    }
    print(f"{name:<22}  {mae_scores.mean():>9.4f}"
          f"  +/-{mae_scores.std():>6.4f}"
          f"  {r2_scores.mean():>8.3f}"
          f"  +/-{r2_scores.std():>6.3f}")

# Plot 7: CV comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
names  = list(cv_results.keys())
colors = ['#5b8db8', '#e07b54', '#6ab187']

ax1.bar(names, [cv_results[n]['mae_mean'] for n in names],
        yerr=[cv_results[n]['mae_std'] for n in names],
        capsize=6, color=colors, alpha=0.85)
ax1.set_ylabel('MAE (log scale)')
ax1.set_title('5-Fold CV: MAE (lower is better)\nError bars = std across folds')
ax1.tick_params(axis='x', rotation=10)

ax2.bar(names, [cv_results[n]['r2_mean'] for n in names],
        yerr=[cv_results[n]['r2_std'] for n in names],
        capsize=6, color=colors, alpha=0.85)
ax2.set_ylabel('R2 Score')
ax2.set_title('5-Fold CV: R2 (higher is better)\nError bars = std across folds')
ax2.tick_params(axis='x', rotation=10)

plt.tight_layout()
plt.savefig('plot7_cv_results.png', dpi=150)
plt.close()
print("CV comparison plot saved.")

# ── 7. GRIDSEARCHCV ON BEST MODEL ────────────────────────────────────────────
best_name = min(cv_results, key=lambda n: cv_results[n]['mae_mean'])
print(f"\nBest model from CV: {best_name}")
print("Running GridSearchCV to tune hyperparameters...")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth':    [3, 5],
    'learning_rate': [0.05, 0.1],
}

grid_search = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=0,
)
grid_search.fit(X_train, y_train_log)

best_params = grid_search.best_params_
best_cv_mae = -grid_search.best_score_
print(f"Best params: {best_params}")
print(f"Best CV MAE (log): {best_cv_mae:.4f}")

# Plot 10: GridSearchCV heatmap (learning_rate vs max_depth, best n_estimators)
best_n = best_params['n_estimators']
results_df = pd.DataFrame(grid_search.cv_results_)
subset = results_df[results_df['param_n_estimators'] == best_n].copy()
subset['lr']    = subset['param_learning_rate'].astype(float)
subset['depth'] = subset['param_max_depth'].astype(int)
pivot = subset.pivot(index='depth', columns='lr', values='mean_test_score')
pivot = -pivot  # convert to MAE

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlOrRd_r', ax=ax)
ax.set_title(f'GridSearchCV MAE (log scale)\nn_estimators={best_n}')
ax.set_xlabel('Learning Rate')
ax.set_ylabel('Max Depth')
plt.tight_layout()
plt.savefig('plot10_gridsearch.png', dpi=150)
plt.close()
print("GridSearchCV plot saved.")

# ── 8. LEARNING CURVES ────────────────────────────────────────────────────────
lc_models = {
    'Linear Regression': models['Linear Regression'],
    'Ridge Regression':  models['Ridge Regression'],
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, random_state=42),
}

train_sizes = np.linspace(0.1, 1.0, 8)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Learning Curves -- Training vs. Validation Error (log-scale MAE)', fontsize=13)

for ax, (name, model) in zip(axes, lc_models.items()):
    sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train_log,
        train_sizes=train_sizes, cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1, shuffle=True, random_state=42,
    )
    train_mae = -train_scores.mean(axis=1)
    val_mae   = -val_scores.mean(axis=1)
    train_std =  train_scores.std(axis=1)
    val_std   =  val_scores.std(axis=1)

    ax.plot(sizes, train_mae, 'o-', color='steelblue', label='Training error')
    ax.plot(sizes, val_mae,   'o-', color='tomato',    label='Validation error')
    ax.fill_between(sizes, train_mae - train_std, train_mae + train_std,
                    alpha=0.15, color='steelblue')
    ax.fill_between(sizes, val_mae - val_std, val_mae + val_std,
                    alpha=0.15, color='tomato')
    ax.set_title(name)
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('MAE (log scale)')
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig('plot8_learning_curves.png', dpi=150)
plt.close()
print("Learning curves saved.")

# ── 9. TRAIN FINAL TUNED MODEL, EVALUATE ON TEST SET ─────────────────────────
final_model = GradientBoostingRegressor(random_state=42, **best_params)
final_model.fit(X_train, y_train_log)

y_pred_log = final_model.predict(X_test)
y_pred     = np.expm1(y_pred_log)       # convert back to dollar scale
test_mae   = mean_absolute_error(y_test, y_pred)
test_r2    = r2_score(y_test, y_pred)
residuals  = y_test - y_pred

print(f"\nFinal model on test set: MAE=${test_mae:,.0f}, R2={test_r2:.3f}")

with open('model.pkl', 'wb') as f:
    pickle.dump(final_model, f)

# Also train the base models for comparison plots
trained_models = {}
for name, model in models.items():
    model.fit(X_train, y_train_log)
    trained_models[name] = model

# Plot 4: Predicted vs Actual
plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred, alpha=0.25, color='steelblue', s=10)
plt.plot([0, 1500], [0, 1500], color='red', linewidth=1.5, label='Perfect prediction')
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.title('Predicted vs Actual Price -- Gradient Boosting (tuned)')
plt.legend()
plt.tight_layout()
plt.savefig('plot4_pred_vs_actual.png', dpi=150)
plt.close()

# Plot 5: Feature coefficients via Ridge Regression
# Features are all scaled by StandardScaler, so coefficient magnitudes are comparable.
ridge = trained_models['Ridge Regression']
coefs = ridge.named_steps['model'].coef_
coef_series = pd.Series(np.abs(coefs), index=feature_labels).sort_values(ascending=True)
plt.figure(figsize=(9, 6))
plt.barh(coef_series.index, coef_series.values, color='steelblue')
plt.xlabel('Absolute Coefficient (standardised features)')
plt.title('Feature Importance via Ridge Coefficients\n(all features scaled -- magnitudes are directly comparable)')
plt.tight_layout()
plt.savefig('plot5_feature_importance.png', dpi=150)
plt.close()

# Shapiro-Wilk normality test on residuals (scipy.stats.shapiro).
# Test uses a random sample because Shapiro-Wilk is designed for n < 5000.
rng = np.random.default_rng(42)
sample_residuals = rng.choice(residuals, size=min(2000, len(residuals)), replace=False)
shapiro_stat, shapiro_p = stats.shapiro(sample_residuals)
print(f"\nShapiro-Wilk test on residuals (n={len(sample_residuals)}): "
      f"W={shapiro_stat:.4f}, p={shapiro_p:.4e}")

# Plot 9: Residual analysis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Residual Analysis -- Gradient Boosting (tuned)', fontsize=13)

ax1.scatter(y_pred, residuals, alpha=0.25, s=10, color='steelblue')
ax1.axhline(0, color='red', linewidth=1.5, linestyle='--')
ax1.set_xlabel('Predicted Price ($)')
ax1.set_ylabel('Residual (Actual - Predicted) ($)')
ax1.set_title('Residuals vs Fitted')

ax2.hist(residuals, bins=60, color='steelblue', edgecolor='white', linewidth=0.4,
         density=True)
ax2.axvline(0, color='red', linestyle='--')
# Overlay fitted normal curve
xr = np.linspace(residuals.min(), residuals.max(), 300)
ax2.plot(xr, stats.norm.pdf(xr, residuals.mean(), residuals.std()),
         color='orange', linewidth=2, label='Normal fit')
sw_label = f'Shapiro-Wilk: W={shapiro_stat:.3f}, p={shapiro_p:.2e}'
ax2.set_xlabel('Residual ($)')
ax2.set_ylabel('Density')
ax2.set_title(f'Residual Distribution\n{sw_label}')
ax2.legend(fontsize=9)

plt.tight_layout()
plt.savefig('plot9_residuals.png', dpi=150)
plt.close()
print("Residual analysis saved.")

# ── 10. SAVE RESULTS ──────────────────────────────────────────────────────────
results = {
    'n_listings':    len(df),
    'best_model':    'Gradient Boosting (tuned)',
    'best_params':   best_params,
    'test_mae':      round(test_mae, 1),
    'test_r2':       round(test_r2, 4),
    'cv_results':    cv_results,
    'mean_price':    round(float(df['price'].mean()), 1),
    'median_price':  round(float(df['price'].median()), 1),
    'n_features':    len(features),
    'shapiro_stat':  round(float(shapiro_stat), 4),
    'shapiro_p':     float(shapiro_p),
    'pearson':       pearson_results,
}

with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nAll plots and results saved.")
print("Run report.py to generate the PDF.")
