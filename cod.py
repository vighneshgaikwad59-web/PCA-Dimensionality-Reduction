import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the breast cancer dataset
def load_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="diagnosis")  # 0 = malignant, 1 = benign
    print(f"[1] Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y, data.target_names

def scale_feature(x):
    scaler= StandardScaler()
    x_scaled = scaler.fit_transform(x)
    print(f"[2] Scaled features")
    return x_scaled, scaler

def apply_pca(x_scaled, n_components=2):# should 2 compnents
    pca=PCA(n_components=n_components)
    x_cpa=pca.fit_transform(x_scaled)
    print(f"[3] pca fir complete ---> reduced {x_scaled.shape[1]} D to {n_components}D")
    return pca,x_cpa

def report_variance(pca):
    """Step 4: How much of the original information (variance) is
    captured by each principal component, and cumulatively."""
    var_ratio = pca.explained_variance_ratio_
    print("[4] Explained variance per component:")
    for i, v in enumerate(var_ratio, start=1):
        print(f"    PC{i}: {v*100:.2f}%")
    print(f"    Total captured by {len(var_ratio)} components: {var_ratio.sum()*100:.2f}%")
    return var_ratio

def plot_project(x_pca,y,target_name,var_ratio,out_path):
    plt.figure(figsize=(8, 6))
    colors = ["#e74c3c", "#2ecc71"]  # malignant, benign
    for label_value,name,color in zip([0,1],target_name,colors):
        mask=y==label_value
        plt.scatter(
       x_pca[mask,0],x_pca[mask,1],
      c=color, label=name, alpha=0.7, edgecolors="k", linewidths=0.3
    )
    plt.xlabel(f"PC1 ({var_ratio[0]*100:.1f}% variance)")
    plt.ylabel(f"PC2 ({var_ratio[1]*100:.1f}% variance)")
    plt.title("PCA: Breast Cancer Dataset (30D -> 2D)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[5] Scatter plot saved -> {out_path}")

def plot_scree(x_scaled,out_path):
   pca_full = PCA().fit(x_scaled)
   cum_var = np.cumsum(pca_full.explained_variance_ratio_)
   plt.figure(figsize=(8, 5))
   plt.plot(range(1, len(cum_var) + 1), cum_var, marker="o", color="#3498db")
   plt.axhline(y=0.95, color="gray", linestyle="--", label="95% variance threshold")
   plt.xlabel("Number of Principal Components")
   plt.ylabel("Cumulative Explained Variance")
   plt.title("Scree Plot: How Many Components Do We Actually Need?")
   plt.legend()
   plt.grid(alpha=0.3)
   plt.tight_layout()
   plt.savefig(out_path, dpi=150)
   print(f"[Bonus] Scree plot saved -> {out_path}")
   n_needed = np.argmax(cum_var>= 0.95) + 1
   print(f"    -> {n_needed} components needed to capture 95% variance")

def main():
    x,y,target_names=load_data()
    x_scaled,scaler =scale_feature(x)
    pca, X_pca = apply_pca(x_scaled, n_components=2)
    var_ratio = report_variance(pca)
    plot_project(X_pca, y, target_names, var_ratio, "pca_projection.png")
    plot_scree(x_scaled, "scree_plot.png")
    print("\nDone. Notice: malignant vs benign samples separate fairly well")
    print("in just 2 dimensions, even though PCA never used the labels.")
 
 
if __name__ == "__main__":
    main()
 
 