#Name:Khushi Srivastava
#Course Code:AER850
#Assignment:Project 1
#Student No. :501126980
#Due Date: October 6,2025
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from scipy.stats import loguniform
import joblib

# Create output folder
OUTDIR = "plots"
os.makedirs(OUTDIR, exist_ok=True)

# Random seed
RANDOM_STATE = 37
np.random.seed(RANDOM_STATE)


plt.style.use("dark_background")
sns.set_theme(style="darkgrid")
plt.rcParams.update({
    "axes.facecolor": "black",
    "figure.facecolor": "black",
    "axes.edgecolor": "#8cb3ff",
    "axes.labelcolor": "#cce0ff",
    "xtick.color": "#a8cfff",
    "ytick.color": "#a8cfff",
    "text.color": "#d0e3ff",
    "grid.color": "#1a1a1a",
    "font.size": 11,
})


# 2.1 Step 1: Data Processing
df = pd.read_csv("Project 1 Data.csv")
df["Step"] = df["Step"].astype(str).str.extract(r"(\d+)").astype(int)
print("Dataset shape:", df.shape)
print(df.head())


# 2.2 Step 2: Data Visualization
print("\nFeature statistics:\n", df[["X", "Y", "Z"]].describe())

# Histograms
df[["X", "Y", "Z"]].hist(figsize=(10, 6), bins=20, color="#083b76", edgecolor="#88bfff")
plt.suptitle("Feature Distributions", fontsize=13, color="#cce0ff")
save_fig("histograms.png")

# Scatter plots
for a, b, t in [("X", "Y", "XY"), ("X", "Z", "XZ"), ("Y", "Z", "YZ")]:
    plt.figure(figsize=(7, 5))
    sc = plt.scatter(df[a], df[b], c=df["Step"], cmap="Blues", edgecolor="#88bfff", alpha=0.85)
    plt.xlabel(a)
    plt.ylabel(b)
    plt.title(f"{a} vs {b} by Step", fontsize=13)
    plt.colorbar(sc).set_label("Step")
    save_fig(f"scatter_{t}.png")


# 2.3 Step 3: Correlation Analysis
corr = df[["X", "Y", "Z", "Step"]].corr(method="pearson")
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f", linewidths=0.4, square=True)
plt.title("Correlation Heatmap", fontsize=13)
save_fig("correlation_heatmap.png")


# 2.4 Step 4: Classification Model Development/Engineering
X = df[["X", "Y", "Z"]]
y = df["Step"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RNG)

num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler())
])
pre = ColumnTransformer([("num", num_pipe, X.columns)], remainder="drop")

dt = Pipeline([("pre", pre), ("clf", DecisionTreeClassifier(random_state=RNG))])
rf = Pipeline([("pre", pre), ("clf", RandomForestClassifier(random_state=RNG))])
svm = Pipeline([("pre", pre), ("clf", SVC(probability=True, random_state=RNG))])
lr = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, random_state=RNG))])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
dt_grid = {"clf__criterion": ["gini", "entropy"], "clf__max_depth": [None, 5, 10, 20]}
rf_grid = {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 10, 20]}
svm_grid = {"clf__kernel": ["rbf", "linear"], "clf__C": [0.1, 1, 10]}
lr_dist = {"clf__C": loguniform(1e-3, 1e2), "clf__solver": ["lbfgs", "saga"]}

dt_search = GridSearchCV(dt, dt_grid, cv=cv, scoring="f1_macro", n_jobs=-1, refit=True)
rf_search = GridSearchCV(rf, rf_grid, cv=cv, scoring="f1_macro", n_jobs=-1, refit=True)
svm_search = GridSearchCV(svm, svm_grid, cv=cv, scoring="f1_macro", n_jobs=-1, refit=True)
lr_search = RandomizedSearchCV(lr, lr_dist, n_iter=20, cv=cv, scoring="f1_macro",
                               random_state=RNG, n_jobs=-1, refit=True)

print("\nTraining models...")
for s in (dt_search, rf_search, svm_search, lr_search):
    s.fit(X_train, y_train)

models = {"Decision Tree": dt_search, "Random Forest": rf_search, "SVM": svm_search, "LogReg": lr_search}


# 2.5 Step 5: Model Performance Analysis
def evaluate(model, X_te, y_te):
    y_pred = model.predict(X_te)
    return {
        "accuracy": accuracy_score(y_te, y_pred),
        "precision": precision_score(y_te, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_te, y_pred, average="macro", zero_division=0)
    }

results = []
for name, s in models.items():
    r = evaluate(s.best_estimator_, X_test, y_test)
    r["model"] = name
    results.append(r)
metrics_df = pd.DataFrame(results).set_index("model").sort_values("f1_macro", ascending=False)
print("\nModel Performance:\n", metrics_df.round(3))
metrics_df.to_csv("model_results.csv")

best_name = metrics_df.index[0]
best_model = models[best_name].best_estimator_


def plot_confusion(model, X_te, y_te, title, fname):
    y_pred = model.predict(X_te)
    cm = confusion_matrix(y_te, y_pred)
    labels = sorted(y_te.unique())
    plt.figure(figsize=(7, 6))
    cmap = sns.color_palette("crest", as_cmap=True)
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, cbar=True, square=True,
                     linewidths=0.6, linecolor="#0d0d0d",
                     xticklabels=labels, yticklabels=labels,
                     annot_kws={"size": 10, "weight": "bold"})
    norm = Normalize(vmin=cm.min(), vmax=cm.max())
    for t in ax.texts:
        v = float(t.get_text())
        r, g, b, _ = cmap(norm(v))
        bright = (0.299 * r + 0.587 * g + 0.114 * b)
        t.set_color("white" if bright < 0.6 else "black")
    plt.title(title, fontsize=13, color="#cce0ff")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    save_fig(fname)
    print(f"\n{title}\n", classification_report(y_te, y_pred, digits=3))


plot_confusion(best_model, X_test, y_test, f"Confusion Matrix {best_name}", "confusion_best.png")


# 2.6 Step 6: Stacked Model Performance Analysis
top2 = metrics_df.head(2).index.tolist()
base_estimators = [(n, models[n].best_estimator_) for n in top2]
stack = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(max_iter=2000, random_state=RNG),
    n_jobs=-1
)
stack.fit(X_train, y_train)
plot_confusion(stack, X_test, y_test, "Confusion Matrix Stacked", "confusion_stacked.png")


# 2.7 Step 7: Model Evaluation
joblib.dump(stack, "final_model.joblib")

coords = np.array([
    [9.375, 3.0625, 1.51],
    [6.995, 5.125, 0.3875],
    [0.0, 3.0625, 1.93],
    [9.4, 3.0, 1.8],
    [9.4, 3.0, 1.3],
])
coords_df = pd.DataFrame(coords, columns=["X", "Y", "Z"])
pred_steps = stack.predict(coords_df)
pred_out = coords_df.copy()
pred_out["Predicted Step"] = pred_steps
print("\nPredictions:\n", pred_out)

plt.figure(figsize=(6, 5))
sc = plt.scatter(pred_out["X"], pred_out["Y"], c=pred_out["Predicted Step"],
                 cmap="Blues", edgecolor="#88bfff", s=70)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Predicted Coordinates by Step", fontsize=13)
plt.colorbar(sc).set_label("Step")
save_fig("predicted_XY.png")

print("\nAll plots saved in 'plots' folder")
