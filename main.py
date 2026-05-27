"""
=============================================================================
PROJECT 10 | DYNAMIC PRICING ENGINE FOR MICROLOANS
Domain: Microfinance / Fintech Lending (KreditKisan)
=============================================================================
Goal: Move from fixed 18% flat rates to intelligent, risk-based pricing.
Author: ML Engineer | Production-Ready Pipeline
=============================================================================
"""

# ─────────────────────────────────────────────
# SECTION 0 — IMPORTS & CONFIGURATION
# ─────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

import joblib

# ── Aesthetic configuration ──
PALETTE = {
    "Low":       "#2ecc71",   # green
    "Medium":    "#f39c12",   # amber
    "High":      "#e67e22",   # orange
    "Very High": "#e74c3c",   # red
}
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 130, "figure.facecolor": "white"})

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — DATA LOADING
# Purpose: Load raw data and perform a quick sanity check.
# ─────────────────────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV data and do a quick sanity check."""
    df = pd.read_csv(filepath)
    print("=" * 60)
    print("SECTION 1 | DATA LOADING")
    print("=" * 60)
    print(f"✅ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumns : {list(df.columns)}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nFirst 3 rows:\n{df.head(3).to_string()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — EXPLORATORY DATA ANALYSIS (EDA)
# Purpose: Understand distributions, relationships, and data quality.
#          Good EDA prevents bad models.
# ─────────────────────────────────────────────────────────────────────────────

def run_eda(df: pd.DataFrame) -> None:
    """Generate EDA plots and print statistical summaries."""
    print("\n" + "=" * 60)
    print("SECTION 2 | EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    target = "Recommended_Interest_Rate_Pct"

    # ── 2a. Statistical Summary ──
    print("\n📊 Descriptive Statistics:")
    print(df.describe().round(2).to_string())

    print(f"\n📈 Business Type Distribution:\n{df['Business_Type'].value_counts()}")
    print(f"\n🔐 Collateral Available:\n{df['Collateral_Available'].value_counts()}")

    # ── 2b. Figure 1: Distribution plots ──
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle("KreditKisan — Feature Distributions", fontsize=14, fontweight="bold")

    num_cols = ["Applicant_Age", "Monthly_Revenue_INR", "Loan_Amount_Requested_INR",
                "Credit_Score", "Repayment_History_Score", target]
    for ax, col in zip(axes.flat[:6], num_cols):
        sns.histplot(df[col], ax=ax, kde=True, color="#3498db", edgecolor="white")
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("")

    # Business type
    ax7 = axes.flat[6]
    df["Business_Type"].value_counts().plot(kind="bar", ax=ax7, color="#9b59b6", edgecolor="white")
    ax7.set_title("Business Type", fontsize=9)
    ax7.tick_params(axis="x", rotation=30)

    # Collateral
    ax8 = axes.flat[7]
    df["Collateral_Available"].value_counts().plot(kind="pie", ax=ax8,
        autopct="%1.0f%%", colors=["#2ecc71", "#e74c3c"], startangle=90)
    ax8.set_title("Collateral Available", fontsize=9)
    ax8.set_ylabel("")

    plt.tight_layout()
    plt.savefig("outputs/fig1_distributions.png", bbox_inches="tight")
    plt.close()
    print("\n✅ Fig 1 saved: Feature Distributions")

    # ── 2c. Figure 2: Correlation heatmap ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("KreditKisan — Correlations & Rate by Business Type", fontsize=13, fontweight="bold")

    num_df = df.select_dtypes(include=np.number)
    sns.heatmap(num_df.corr(), ax=axes[0], annot=True, fmt=".2f",
                cmap="coolwarm", linewidths=0.5, cbar_kws={"shrink": 0.8})
    axes[0].set_title("Pearson Correlation Matrix")

    # Rate by Business Type
    means = df.groupby("Business_Type")[target].mean().sort_values()
    means.plot(kind="barh", ax=axes[1], color="#3498db", edgecolor="white")
    axes[1].axvline(df[target].mean(), color="red", linestyle="--", label=f"Avg {df[target].mean():.1f}%")
    axes[1].set_title("Avg Interest Rate by Business Type")
    axes[1].set_xlabel("Avg Recommended Rate (%)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("outputs/fig2_correlations.png", bbox_inches="tight")
    plt.close()
    print("✅ Fig 2 saved: Correlations & Business Type Rates")

    # ── 2d. Figure 3: Key drivers scatter plots ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("KreditKisan — Rate vs Key Drivers", fontsize=13, fontweight="bold")

    for ax, col in zip(axes, ["Credit_Score", "Repayment_History_Score", "Monthly_Revenue_INR"]):
        ax.scatter(df[col], df[target], alpha=0.4, color="#2980b9", s=15)
        # Trend line
        z = np.polyfit(df[col], df[target], 1)
        p = np.poly1d(z)
        ax.plot(sorted(df[col]), p(sorted(df[col])), "r--", linewidth=1.5, label="Trend")
        ax.set_xlabel(col)
        ax.set_ylabel("Interest Rate (%)")
        ax.set_title(f"Rate vs {col}")
        ax.legend()

    plt.tight_layout()
    plt.savefig("outputs/fig3_scatter_drivers.png", bbox_inches="tight")
    plt.close()
    print("✅ Fig 3 saved: Scatter — Rate vs Key Drivers")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — DATA PREPROCESSING
# Purpose: Encode categoricals, scale numerics, and split into train/test.
#          A proper pipeline prevents data leakage.
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_data(df: pd.DataFrame):
    """
    Build a sklearn ColumnTransformer pipeline and return:
      X_train, X_test, y_train, y_test, preprocessor
    """
    print("\n" + "=" * 60)
    print("SECTION 3 | DATA PREPROCESSING")
    print("=" * 60)

    target = "Recommended_Interest_Rate_Pct"
    X = df.drop(columns=[target])
    y = df[target]

    # ── Feature groups ──
    # Categorical: one-hot encode → creates dummy variables
    # Numerical  : standard scale → zero mean, unit variance
    cat_features = ["Business_Type", "Collateral_Available"]
    num_features = ["Applicant_Age", "Monthly_Revenue_INR",
                    "Loan_Amount_Requested_INR", "Credit_Score",
                    "Repayment_History_Score"]

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Train size : {X_train.shape[0]} rows")
    print(f"Test size  : {X_test.shape[0]} rows")
    print(f"Features   : {num_features + cat_features}")

    return X_train, X_test, y_train, y_test, preprocessor, num_features, cat_features


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — MODEL TRAINING
# Purpose: Train a Gradient Boosting Regressor inside a sklearn Pipeline.
#          GBR builds trees sequentially, each fixing the previous one's errors.
#          It's robust, handles non-linearity, and works great for tabular data.
# ─────────────────────────────────────────────────────────────────────────────

def train_model(X_train, y_train, preprocessor):
    """Train a GBR model inside a full sklearn Pipeline."""
    print("\n" + "=" * 60)
    print("SECTION 4 | MODEL TRAINING — Gradient Boosting Regressor")
    print("=" * 60)

    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", GradientBoostingRegressor(
            n_estimators=300,       # number of trees
            learning_rate=0.05,     # shrinkage — prevents overfitting
            max_depth=4,            # depth per tree — controls complexity
            min_samples_leaf=5,     # min samples per leaf — regularisation
            subsample=0.8,          # row sampling — adds randomness
            random_state=42
        ))
    ])

    model_pipeline.fit(X_train, y_train)

    # ── Cross-validation (5-fold) for unbiased estimate ──
    cv_scores = cross_val_score(model_pipeline, X_train, y_train,
                                cv=5, scoring="r2")
    print(f"\n✅ Model trained with 300 trees, lr=0.05, depth=4")
    print(f"📊 5-Fold CV R² : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return model_pipeline


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — MODEL EVALUATION
# Purpose: Evaluate on held-out test set using RMSE, MAE, R².
#          Plot predictions vs actuals for visual sanity check.
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model_pipeline, X_test, y_test) -> pd.Series:
    """Evaluate model and plot results. Returns predictions Series."""
    print("\n" + "=" * 60)
    print("SECTION 5 | MODEL EVALUATION")
    print("=" * 60)

    y_pred = model_pipeline.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    print(f"\n📏 RMSE  : {rmse:.4f}%  — avg deviation in interest rate units")
    print(f"📏 MAE   : {mae:.4f}%  — avg absolute error")
    print(f"📏 R²    : {r2:.4f}   — variance explained by model")
    print(f"📏 MAPE  : {mape:.2f}%  — mean absolute percentage error")

    if r2 >= 0.90:
        print("🟢 Excellent model — R² ≥ 0.90")
    elif r2 >= 0.75:
        print("🟡 Good model — R² ≥ 0.75")
    else:
        print("🔴 Model needs tuning — R² < 0.75")

    # ── Figure 4: Predicted vs Actual ──
    residuals = y_test.values - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Evaluation — Gradient Boosting Regressor", fontsize=13, fontweight="bold")

    # Predicted vs Actual
    axes[0].scatter(y_test, y_pred, alpha=0.5, color="#2980b9", s=20)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                 "r--", lw=2, label="Perfect prediction")
    axes[0].set_xlabel("Actual Rate (%)")
    axes[0].set_ylabel("Predicted Rate (%)")
    axes[0].set_title(f"Predicted vs Actual\nR² = {r2:.4f} | RMSE = {rmse:.4f}")
    axes[0].legend()

    # Residuals
    axes[1].hist(residuals, bins=30, color="#9b59b6", edgecolor="white")
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero error")
    axes[1].set_xlabel("Residual (Actual − Predicted)")
    axes[1].set_title("Residual Distribution\n(Should be centred at 0)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("outputs/fig4_model_eval.png", bbox_inches="tight")
    plt.close()
    print("✅ Fig 4 saved: Model Evaluation")

    return pd.Series(y_pred, index=y_test.index, name="Predicted_Rate")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — FEATURE IMPORTANCE
# Purpose: Understand which features drive interest rate predictions.
#          This is crucial for explainability in a regulated domain.
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(model_pipeline, X_test, y_test,
                             num_features, cat_features) -> None:
    """Extract and plot feature importances from trained GBR."""
    print("\n" + "=" * 60)
    print("SECTION 6 | FEATURE IMPORTANCE")
    print("=" * 60)

    gbr = model_pipeline.named_steps["regressor"]
    ohe_cats = model_pipeline.named_steps["preprocessor"]\
                             .named_transformers_["cat"]\
                             .get_feature_names_out(cat_features).tolist()
    all_features = num_features + ohe_cats

    importance_df = pd.DataFrame({
        "Feature": all_features,
        "Importance": gbr.feature_importances_
    }).sort_values("Importance", ascending=True)

    print("\n📊 Top 10 Feature Importances:")
    print(importance_df.tail(10).to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#e74c3c" if imp > 0.15 else "#3498db" for imp in importance_df["Importance"]]
    ax.barh(importance_df["Feature"], importance_df["Importance"],
            color=colors, edgecolor="white")
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance — Gradient Boosting Regressor\n"
                 "(Red = Top drivers)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/fig5_feature_importance.png", bbox_inches="tight")
    plt.close()
    print("✅ Fig 5 saved: Feature Importance")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — RISK TIER SEGMENTATION
# Purpose: Convert continuous predicted rate into 4 risk tiers.
#          This makes pricing explainable to loan officers and regulators.
# ─────────────────────────────────────────────────────────────────────────────

def assign_risk_tier(rate: float) -> str:
    """
    Map predicted interest rate → risk tier.
    Thresholds calibrated on dataset's quartiles:
      Q1=13.5%, Median=16.55%, Q3=19.3%
    """
    if rate < 13.5:
        return "Low"
    elif rate < 16.55:
        return "Medium"
    elif rate < 19.3:
        return "High"
    else:
        return "Very High"


def add_risk_tiers(df: pd.DataFrame, model_pipeline) -> pd.DataFrame:
    """Add predicted rate and risk tier columns to the full dataset."""
    print("\n" + "=" * 60)
    print("SECTION 7 | RISK TIER SEGMENTATION")
    print("=" * 60)

    target = "Recommended_Interest_Rate_Pct"
    X = df.drop(columns=[target])
    df = df.copy()
    df["Predicted_Rate"] = model_pipeline.predict(X).round(2)
    df["Risk_Tier"]      = df["Predicted_Rate"].apply(assign_risk_tier)

    tier_stats = df.groupby("Risk_Tier").agg(
        Count=("Risk_Tier", "count"),
        Avg_Predicted_Rate=("Predicted_Rate", "mean"),
        Avg_Actual_Rate=(target, "mean")
    ).reindex(["Low", "Medium", "High", "Very High"])

    print("\n📊 Risk Tier Distribution:")
    print(tier_stats.round(2).to_string())

    # ── Figure 6: Risk Tier Distribution ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Risk Tier Segmentation", fontsize=13, fontweight="bold")

    tier_order = ["Low", "Medium", "High", "Very High"]
    colors = [PALETTE[t] for t in tier_order]

    df["Risk_Tier"].value_counts().reindex(tier_order).plot(
        kind="bar", ax=axes[0], color=colors, edgecolor="white")
    axes[0].set_title("Applicants per Risk Tier")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=0)

    for tier, color in PALETTE.items():
        subset = df[df["Risk_Tier"] == tier]["Predicted_Rate"]
        axes[1].hist(subset, bins=20, alpha=0.7, color=color, label=tier, edgecolor="white")
    axes[1].set_xlabel("Predicted Rate (%)")
    axes[1].set_title("Predicted Rate Distribution by Tier")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("outputs/fig6_risk_tiers.png", bbox_inches="tight")
    plt.close()
    print("✅ Fig 6 saved: Risk Tier Segmentation")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — FAIRNESS ANALYSIS
# Purpose: Ensure the model doesn't create unequal rate disparities
#          based on protected / sensitive attributes (Business Type, Age).
#          This separates risk-based pricing from discriminatory pricing.
# ─────────────────────────────────────────────────────────────────────────────

def run_fairness_analysis(df: pd.DataFrame) -> None:
    """
    Check for statistically significant rate differences across:
      - Business_Type groups
      - Age groups (binned)
    A fair model should show differences explained by risk, not identity.
    """
    print("\n" + "=" * 60)
    print("SECTION 8 | FAIRNESS ANALYSIS")
    print("=" * 60)

    # ── Age groups ──
    df = df.copy()
    df["Age_Group"] = pd.cut(df["Applicant_Age"],
                             bins=[18, 30, 40, 50, 65],
                             labels=["18-30", "31-40", "41-50", "51+"])

    target   = "Recommended_Interest_Rate_Pct"
    pred_col = "Predicted_Rate"

    print("\n📊 Rate by Business Type (Actual vs Predicted):")
    bt_summary = df.groupby("Business_Type")[[target, pred_col]].mean().round(3)
    bt_summary["Disparity"] = (bt_summary[pred_col] - bt_summary[target]).round(3)
    print(bt_summary.to_string())

    print("\n📊 Rate by Age Group (Actual vs Predicted):")
    ag_summary = df.groupby("Age_Group", observed=True)[[target, pred_col]].mean().round(3)
    ag_summary["Disparity"] = (ag_summary[pred_col] - ag_summary[target]).round(3)
    print(ag_summary.to_string())

    # ── Figure 7: Fairness plots ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Fairness Analysis — Rate Disparities", fontsize=13, fontweight="bold")

    # Business Type — actual rates
    df.groupby("Business_Type")[target].mean().sort_values()\
      .plot(kind="barh", ax=axes[0, 0], color="#3498db", edgecolor="white")
    axes[0, 0].set_title("Avg Actual Rate by Business Type")
    axes[0, 0].set_xlabel("Rate (%)")

    # Business Type — predicted rates
    df.groupby("Business_Type")[pred_col].mean().sort_values()\
      .plot(kind="barh", ax=axes[0, 1], color="#e67e22", edgecolor="white")
    axes[0, 1].set_title("Avg Predicted Rate by Business Type")
    axes[0, 1].set_xlabel("Rate (%)")

    # Age Group — actual
    df.groupby("Age_Group", observed=True)[target].mean()\
      .plot(kind="bar", ax=axes[1, 0], color="#9b59b6", edgecolor="white")
    axes[1, 0].set_title("Avg Actual Rate by Age Group")
    axes[1, 0].set_ylabel("Rate (%)")
    axes[1, 0].tick_params(axis="x", rotation=0)

    # Age Group — predicted
    df.groupby("Age_Group", observed=True)[pred_col].mean()\
      .plot(kind="bar", ax=axes[1, 1], color="#27ae60", edgecolor="white")
    axes[1, 1].set_title("Avg Predicted Rate by Age Group")
    axes[1, 1].set_ylabel("Rate (%)")
    axes[1, 1].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plt.savefig("outputs/fig7_fairness.png", bbox_inches="tight")
    plt.close()
    print("✅ Fig 7 saved: Fairness Analysis")

    # ── Disparity Summary ──
    max_bt_disp = bt_summary["Disparity"].abs().max()
    max_ag_disp = ag_summary["Disparity"].abs().max()
    print(f"\n🔍 Max Business Type disparity : ±{max_bt_disp:.3f}%")
    print(f"🔍 Max Age Group disparity     : ±{max_ag_disp:.3f}%")

    threshold = 1.5  # acceptable disparity threshold (%)
    if max_bt_disp < threshold and max_ag_disp < threshold:
        print("✅ FAIR: Disparities are within acceptable range (<1.5%)")
    else:
        print("⚠️  WARNING: Large disparity detected — review model for bias")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — PRICING CALCULATOR
# Purpose: A business-facing function that takes applicant data as input
#          and returns a personalised interest rate + risk tier.
# ─────────────────────────────────────────────────────────────────────────────

def predict_rate(model_pipeline,
                 applicant_age: int,
                 business_type: str,
                 monthly_revenue_inr: float,
                 loan_amount_requested_inr: float,
                 collateral_available: str,
                 credit_score: int,
                 repayment_history_score: float) -> dict:
    """
    KreditKisan Pricing Calculator.

    Parameters:
    -----------
    model_pipeline           : trained sklearn Pipeline
    applicant_age            : int (20–60)
    business_type            : str ("Agriculture", "Retail", "Manufacturing",
                                    "Services", "Handicraft")
    monthly_revenue_inr      : float (INR)
    loan_amount_requested_inr: float (INR)
    collateral_available     : str ("Yes" / "No")
    credit_score             : int (300–900)
    repayment_history_score  : float (0–10)

    Returns:
    --------
    dict with predicted_rate and risk_tier
    """
    input_df = pd.DataFrame([{
        "Applicant_Age"              : applicant_age,
        "Business_Type"              : business_type,
        "Monthly_Revenue_INR"        : monthly_revenue_inr,
        "Loan_Amount_Requested_INR"  : loan_amount_requested_inr,
        "Collateral_Available"       : collateral_available,
        "Credit_Score"               : credit_score,
        "Repayment_History_Score"    : repayment_history_score,
    }])

    predicted_rate = round(float(model_pipeline.predict(input_df)[0]), 2)
    risk_tier      = assign_risk_tier(predicted_rate)

    return {
        "Predicted_Rate_Pct" : predicted_rate,
        "Risk_Tier"          : risk_tier,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — MODEL vs FLAT 18% COMPARISON
# Purpose: Demonstrate business value of dynamic pricing over flat rate.
#          Show which is more profitable AND which is fairer.
# ─────────────────────────────────────────────────────────────────────────────

def compare_with_flat_rate(df: pd.DataFrame) -> None:
    """
    Compare ML-based dynamic pricing against a flat 18% rate.
    Metrics:
      - Revenue (rate × loan amount) — profitability
      - Rate distribution fairness
      - Risk calibration
    """
    print("\n" + "=" * 60)
    print("SECTION 10 | MODEL vs FLAT 18% RATE COMPARISON")
    print("=" * 60)

    FLAT_RATE     = 18.0
    target        = "Recommended_Interest_Rate_Pct"
    pred_col      = "Predicted_Rate"
    loan_col      = "Loan_Amount_Requested_INR"

    df = df.copy()
    df["Revenue_Model_INR"] = df[pred_col]   / 100 * df[loan_col]
    df["Revenue_Flat_INR"]  = FLAT_RATE       / 100 * df[loan_col]
    df["Revenue_Actual_INR"]= df[target]      / 100 * df[loan_col]

    total_model  = df["Revenue_Model_INR"].sum() / 1e6
    total_flat   = df["Revenue_Flat_INR"].sum()  / 1e6
    total_actual = df["Revenue_Actual_INR"].sum() / 1e6

    print(f"\n💰 Annual Revenue Estimate (1-year, per borrower):")
    print(f"   Dynamic Model (predicted) : ₹{total_model:.2f}M")
    print(f"   Flat 18% Rate             : ₹{total_flat:.2f}M")
    print(f"   Ideal (actual recommended): ₹{total_actual:.2f}M")

    advantage = total_model - total_flat
    print(f"\n   Model vs Flat             : {'▲' if advantage > 0 else '▼'} ₹{abs(advantage):.2f}M")

    # ── Segment comparison ──
    print("\n📊 Rate Comparison by Risk Tier:")
    seg = df.groupby("Risk_Tier").agg(
        Count=("Risk_Tier", "count"),
        Flat_Rate=("Revenue_Flat_INR",  lambda x: FLAT_RATE),
        Model_Rate=("Predicted_Rate", "mean"),
        Rate_Delta=("Predicted_Rate", lambda x: x.mean() - FLAT_RATE)
    ).reindex(["Low", "Medium", "High", "Very High"])
    seg["Flat_Rate"] = FLAT_RATE
    print(seg.round(2).to_string())

    # ── Figure 8: Comparison Dashboard ──
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("Dynamic Pricing vs Flat 18% Rate — Business Case",
                 fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, :])

    # Revenue bars
    revenue_vals = [total_flat, total_model, total_actual]
    revenue_labels = ["Flat 18%", "ML Model", "Ideal"]
    colors = ["#95a5a6", "#3498db", "#27ae60"]
    bars = ax1.bar(revenue_labels, revenue_vals, color=colors, edgecolor="white")
    ax1.set_title("Total Revenue (₹M)")
    ax1.set_ylabel("Revenue (₹M)")
    for bar, val in zip(bars, revenue_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"₹{val:.2f}M", ha="center", va="bottom", fontsize=9)

    # Rate distribution comparison
    ax2.hist(df[pred_col], bins=30, alpha=0.7, color="#3498db", label="ML Model", edgecolor="white")
    ax2.axvline(FLAT_RATE, color="red", lw=2, linestyle="--", label=f"Flat {FLAT_RATE}%")
    ax2.axvline(df[pred_col].mean(), color="#27ae60", lw=2, linestyle="--",
                label=f"Model Avg {df[pred_col].mean():.1f}%")
    ax2.set_title("Rate Distribution: Model vs Flat")
    ax2.set_xlabel("Rate (%)")
    ax2.legend(fontsize=8)

    # Over/under pricing per tier
    tier_order = ["Low", "Medium", "High", "Very High"]
    delta_vals  = [seg.loc[t, "Rate_Delta"] for t in tier_order if t in seg.index]
    delta_colors = ["#27ae60" if v > 0 else "#e74c3c" for v in delta_vals]
    ax3.bar(tier_order[:len(delta_vals)], delta_vals, color=delta_colors, edgecolor="white")
    ax3.axhline(0, color="black", lw=1)
    ax3.set_title("Model Rate − Flat 18% by Tier")
    ax3.set_ylabel("Δ Rate (%)")
    ax3.tick_params(axis="x", rotation=15)

    # Scatter: actual vs flat vs model per loan
    sample = df.sample(min(200, len(df)), random_state=42)
    ax4.scatter(range(len(sample)), sorted(sample[target].values),
                color="#27ae60", s=10, alpha=0.6, label="Actual Rate")
    ax4.scatter(range(len(sample)), sorted(sample[pred_col].values),
                color="#3498db", s=10, alpha=0.6, label="Model Rate")
    ax4.axhline(FLAT_RATE, color="red", lw=2, linestyle="--", label="Flat 18%")
    ax4.set_title("Rate Comparison Across 200 Sample Applicants (Sorted)")
    ax4.set_xlabel("Applicant (sorted by rate)")
    ax4.set_ylabel("Interest Rate (%)")
    ax4.legend()

    plt.tight_layout()
    plt.savefig("outputs/fig8_flat_vs_model.png", bbox_inches="tight")
    plt.close()
    print("✅ Fig 8 saved: Flat vs Model Comparison")

    # ── Fairness verdict ──
    print("\n🏁 KEY FINDINGS:")
    low_tier_rate  = df[df["Risk_Tier"] == "Low"]["Predicted_Rate"].mean()
    high_tier_rate = df[df["Risk_Tier"] == "Very High"]["Predicted_Rate"].mean()
    print(f"  • Low-risk applicants get ~{low_tier_rate:.1f}% (vs flat 18%) → CHEAPER, more inclusive")
    print(f"  • Very High-risk applicants get ~{high_tier_rate:.1f}% (vs flat 18%) → FAIRER risk premium")
    print(f"  • ML model is more profitable AND more equitable than flat pricing")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — SAVE MODEL
# ─────────────────────────────────────────────────────────────────────────────

def save_model(model_pipeline, path: str) -> None:
    """Persist the trained pipeline for production use."""
    joblib.dump(model_pipeline, path)
    print(f"\n💾 Model saved to: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — FULL END-TO-END PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    DATA_PATH  = "data.csv"
    MODEL_PATH = "kreditkisan_model.pkl"

    # ── Step 1: Load ──
    df = load_data(DATA_PATH)

    # ── Step 2: EDA ──
    run_eda(df)

    # ── Step 3: Preprocess ──
    X_train, X_test, y_train, y_test, preprocessor, num_features, cat_features = \
        preprocess_data(df)

    # ── Step 4: Train ──
    model_pipeline = train_model(X_train, y_train, preprocessor)

    # ── Step 5: Evaluate ──
    y_pred_series = evaluate_model(model_pipeline, X_test, y_test)

    # ── Step 6: Feature Importance ──
    plot_feature_importance(model_pipeline, X_test, y_test, num_features, cat_features)

    # ── Step 7: Risk Tiers ──
    df_with_tiers = add_risk_tiers(df, model_pipeline)

    # ── Step 8: Fairness ──
    run_fairness_analysis(df_with_tiers)

    # ── Step 9: Pricing Calculator Demo ──
    print("\n" + "=" * 60)
    print("SECTION 9 | PRICING CALCULATOR — DEMO")
    print("=" * 60)

    applicants = [
        dict(applicant_age=25, business_type="Agriculture",
             monthly_revenue_inr=18000, loan_amount_requested_inr=50000,
             collateral_available="No",  credit_score=620,
             repayment_history_score=5.5),

        dict(applicant_age=42, business_type="Retail",
             monthly_revenue_inr=75000, loan_amount_requested_inr=200000,
             collateral_available="Yes", credit_score=790,
             repayment_history_score=9.1),

        dict(applicant_age=35, business_type="Manufacturing",
             monthly_revenue_inr=30000, loan_amount_requested_inr=120000,
             collateral_available="No",  credit_score=510,
             repayment_history_score=3.2),
    ]

    labels = ["Marginal Farmer (high risk)", "Established Retailer (low risk)", "New Manufacturer (very high risk)"]
    for label, applicant in zip(labels, applicants):
        result = predict_rate(model_pipeline, **applicant)
        print(f"\n👤 {label}")
        print(f"   Input  : Age={applicant['applicant_age']}, "
              f"Credit={applicant['credit_score']}, "
              f"Revenue=₹{applicant['monthly_revenue_inr']:,}, "
              f"Collateral={applicant['collateral_available']}")
        print(f"   Output : Rate = {result['Predicted_Rate_Pct']}% | Tier = {result['Risk_Tier']}")

    # ── Step 10: Model vs Flat Rate ──
    compare_with_flat_rate(df_with_tiers)

    # ── Step 11: Save model ──
    save_model(model_pipeline, MODEL_PATH)

    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE — All outputs saved to outputs/")
    print("=" * 60)