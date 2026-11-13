from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

def save_eda_figures(df, X, figures_dir):
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Target distribution
    target_col = next((c for c in df.columns if c.lower() in
                       ["potability", "target", "is_safe", "water_quality"]), None)
    if target_col:
        plt.figure(figsize=(6, 4))
        sns.countplot(x=df[target_col])
        plt.title("Target Class Distribution")
        plt.xlabel(target_col)
        plt.ylabel("Number of samples")
        plt.tight_layout()
        plt.savefig(figures_dir / "target_distribution.png", dpi=150)
        plt.close()

    # Correlation heatmap
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] >= 2:
        plt.figure(figsize=(10, 7))
        sns.heatmap(numeric.corr(), cmap="coolwarm", center=0, annot=False)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(figures_dir / "correlation_heatmap.png", dpi=150)
        plt.close()

    # Feature distributions
    if X.shape[1] > 0:
        cols = list(X.columns[:9])
        X[cols].hist(figsize=(12, 9), bins=25)
        plt.suptitle("Feature Distributions")
        plt.tight_layout()
        plt.savefig(figures_dir / "feature_distributions.png", dpi=150)
        plt.close()

def save_evaluation_figures(model, X_test, y_test, figures_dir):
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(figures_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    classifier = model.named_steps["classifier"]
    importances = classifier.feature_importances_
    names = X_test.columns
    order = importances.argsort()

    plt.figure(figsize=(8, 5))
    plt.barh(names[order], importances[order])
    plt.xlabel("Importance")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(figures_dir / "feature_importance.png", dpi=150)
    plt.close()
