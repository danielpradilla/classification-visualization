# Classification Model Visualizer

Interactive static site for explaining supervised classification and unsupervised clustering models.

## Project note

This project is a small interactive explainer for seeing how different classification and clustering approaches behave on the same two-dimensional data. The goal is not to reproduce production machine-learning libraries. It is to make model behavior visible: where a boundary is linear, where it bends, where it overfits noisy points, and where a clustering method treats sparse regions as noise.

The app has two views. The single model view is for teaching one method at a time, with model-specific controls and explanatory notes. The Classification Model Wall puts every model in a small-multiple layout so the same generated dataset can be compared side by side. That makes it easier to see which methods pick up the intended pattern and which ones fail because their assumptions do not match the data shape.

All datasets are generated synthetically in the browser. The generator creates familiar teaching shapes such as separated blobs, overlapping blobs, imbalanced classes, anisotropic blobs, XOR quadrants, interlocking moons, nested rings, twin spirals, cluster islands, concentric blobs, bridge/noisy-connector clusters, elongated clusters, outlier-heavy clusters, variable-size clusters, variable-density clusters, and curved chains. The sample count and noise controls change the generated points, while a deterministic seed keeps each model in a comparison view training on the same data.

The implementation is plain static HTML, CSS, and JavaScript with D3 for drawing. Model metadata, dataset names, palette colors, and shared sorting helpers live in `shared-data.js`; common visual tokens and navigation styles live in `shared.css`. The visual styleguide is available as `styleguide.html`.

## What is included

- Supervised models: logistic regression, k-nearest neighbors, linear SVM, RBF kernel SVM, neural network, LDA, QDA, decision tree, random forest, XGBoost-style boosted trees, Gaussian naive Bayes.
- Unsupervised models: k-means, DBSCAN, spectral clustering, OPTICS/HDBSCAN-style density clustering, Gaussian mixture, agglomerative clustering.
- Synthetic datasets with adjustable sample count and noise.
- Dataset shapes include overlapping, imbalanced, anisotropic, XOR, spiral, moons, nested rings, concentric blobs, noisy connectors, elongated clusters, outlier-heavy clusters, variable density, and variable size clusters.
- Visual decision regions, cluster assignments, centroids, graph links, density rings, support vectors, tree splits, boosted-tree corrections, and holdout accuracy.
- Wikipedia links for each model.
- Model-specific controls show and hide automatically: regularization, kernel width, hidden units, learning rates, epochs, tree counts, boosting rounds, GMM shape/assignment, graph neighbors, and minimum cluster size.
- A Classification Model Wall in `compare.html`.
- A visual styleguide in `styleguide.html`.

## Run locally

```bash
python3 -m http.server 5173
```

Then open `http://localhost:5173`.


