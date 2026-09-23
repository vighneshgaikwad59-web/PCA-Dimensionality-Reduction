# 🧬 PCA Dimensionality Reduction — Breast Cancer Dataset

Reduces 30 numeric cell-nuclei features (from the sklearn Breast Cancer
Wisconsin dataset) down to 2 principal components using **PCA**, and
visualizes whether malignant vs. benign samples still separate cleanly
after the reduction.

## 🔍 Why PCA?

Each sample has 30 measurements (radius, texture, smoothness, concavity...).
That's too many dimensions to plot or reason about directly. PCA finds the
directions in which the data varies the most and projects everything onto
just a couple of those directions — like compressing a full lab panel into
2 headline indicators that still tell most of the story.

## ⚙️ Pipeline

1. **Load** — `sklearn.datasets.load_breast_cancer` (569 samples, 30 features)
2. **Scale** — `StandardScaler` (PCA is scale-sensitive, so every feature
   is normalized to mean 0, std 1 first)
3. **Reduce** — `PCA(n_components=2)` → 30D to 2D
4. **Report** — explained variance ratio per component
5. **Visualize** — 2D scatter plot colored by actual diagnosis
6. **Bonus** — scree plot showing cumulative variance across *all*
   30 components, to justify how many you'd actually need

## 📊 Results

| Component | Variance Explained |
|-----------|--------------------|
| PC1       | ~44%               |
| PC2       | ~19%               |
| **Total (2D)** | **~63%**      |

~10 components are needed to retain 95% of the original variance — but
even just 2 components separate malignant vs. benign fairly well visually,
which is the whole point of PCA as an exploration/visualization tool.

## 🚀 Run it

```bash
pip install -r requirements.txt
python pca_analysis.py
```

Generates `pca_projection.png` (2D class separation) and `scree_plot.png`
(variance vs. number of components) in the project folder.

## 🛠️ Tech Stack

- scikit-learn (`PCA`, `StandardScaler`)
- pandas / numpy
- matplotlib

## 💡 Next Steps / Extensions

- Try `n_components=3` and plot in 3D
- Feed the 2 PCA components into a classifier (Logistic Regression / KNN)
  and compare accuracy vs. using all 30 raw features
- Look at `pca.components_` to see which original features load most
  heavily onto PC1 and PC2
  
