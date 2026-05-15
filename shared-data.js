(() => {
const sharedPalette = [
  "#2563eb",
  "#c2413f",
  "#0f766e",
  "#b7791f",
  "#7c3aed",
  "#374151",
  "#db2777",
  "#65a30d",
  "#0891b2",
  "#ea580c",
  "#4f46e5",
  "#525252",
];
const sharedNoiseColor = "#6b7280";

const modelCatalog = {
  supervised: [
    {
      id: "logistic",
      name: "Logistic regression",
      subtitle: "A linear separator learned from labeled examples.",
      complexity: "Low",
      notes: [
        "Fits one smooth straight boundary between classes.",
        "Turns distance from the boundary into a probability.",
        "Struggles when the real pattern bends or wraps around itself.",
      ],
      useCase: "Baseline binary classification, interpretable scoring, and cases where features separate classes roughly linearly.",
      wiki: "https://en.wikipedia.org/wiki/Logistic_regression",
    },
    {
      id: "knn",
      name: "k-nearest neighbors",
      subtitle: "A local vote from the nearest labeled examples.",
      complexity: "Medium",
      notes: [
        "Stores the training examples instead of learning a compact formula.",
        "Classifies each new point by nearby labels.",
        "Small k follows noise; large k smooths away local structure.",
      ],
      useCase: "Intuitive prototypes, recommendation-style similarity problems, and datasets where local neighborhoods are meaningful.",
      wiki: "https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm",
    },
    {
      id: "svm",
      name: "Linear SVM",
      subtitle: "A wide-margin line that tries to stay away from both classes.",
      complexity: "Low",
      notes: [
        "Prioritizes the points closest to the boundary.",
        "Optimizes for margin, not calibrated probability.",
        "A kernel SVM can bend; this simplified view shows the linear core.",
      ],
      useCase: "High-dimensional classification, text features, and problems where a stable separating margin matters.",
      wiki: "https://en.wikipedia.org/wiki/Support_vector_machine",
    },
    {
      id: "rbf-svm",
      name: "RBF kernel SVM",
      subtitle: "A margin classifier whose boundary bends through a radial kernel.",
      complexity: "High",
      notes: [
        "Compares points by distance instead of drawing one straight line.",
        "Support vectors shape local bends in the boundary.",
        "The kernel width controls whether the model is smooth or wiggly.",
      ],
      useCase: "Nonlinear binary classification when curved boundaries matter, especially moons, rings, and compact class islands.",
      wiki: "https://en.wikipedia.org/wiki/Radial_basis_function_kernel",
    },
    {
      id: "neural-net",
      name: "Neural network",
      subtitle: "Hidden units combine soft nonlinear cuts into a flexible boundary.",
      complexity: "High",
      notes: [
        "Learns several hidden features from x and y.",
        "Each training step nudges weights to reduce classification error.",
        "More capacity can fit curved patterns, but it can also chase noise.",
      ],
      useCase: "Flexible nonlinear classification, representation learning, and problems where feature interactions are hard to specify manually.",
      wiki: "https://en.wikipedia.org/wiki/Artificial_neural_network",
    },
    {
      id: "lda",
      name: "Linear discriminant analysis",
      subtitle: "Class Gaussians with one shared covariance create a linear boundary.",
      complexity: "Low",
      notes: [
        "Estimates a center for each class.",
        "Assumes both classes share the same covariance shape.",
        "The shared shape keeps the boundary linear and fairly stable.",
      ],
      useCase: "Small datasets, interpretable linear baselines, and data where each class is roughly Gaussian with similar spread.",
      wiki: "https://en.wikipedia.org/wiki/Linear_discriminant_analysis",
    },
    {
      id: "qda",
      name: "Quadratic discriminant analysis",
      subtitle: "Each class gets its own covariance, so the boundary can curve.",
      complexity: "Medium",
      notes: [
        "Estimates a separate ellipse-shaped distribution for each class.",
        "Allows curved decision boundaries when class spreads differ.",
        "Needs more data than LDA because it estimates more parameters.",
      ],
      useCase: "Probabilistic classification where classes have different variance, orientation, or elliptical shapes.",
      wiki: "https://en.wikipedia.org/wiki/Quadratic_classifier#Quadratic_discriminant_analysis",
    },
    {
      id: "tree",
      name: "Decision tree",
      subtitle: "A sequence of axis-aligned if/then splits.",
      complexity: "Medium",
      notes: [
        "Cuts the space into rectangles.",
        "Each split asks a single feature question.",
        "More depth captures more detail but can memorize noise.",
      ],
      useCase: "Readable rules, mixed feature types, and workflows where explanations need to be close to business logic.",
      wiki: "https://en.wikipedia.org/wiki/Decision_tree_learning",
    },
    {
      id: "forest",
      name: "Random forest",
      subtitle: "Many noisy trees voting together.",
      complexity: "High",
      notes: [
        "Builds several trees on random subsets of rows and features.",
        "Averaging votes reduces single-tree overfitting.",
        "The boundary can become flexible while staying fairly stable.",
      ],
      useCase: "Strong tabular-data baselines, nonlinear feature interactions, and practical classification with limited tuning.",
      wiki: "https://en.wikipedia.org/wiki/Random_forest",
    },
    {
      id: "xgboost",
      name: "XGBoost-style boosted trees",
      subtitle: "Shallow trees added one after another to fix prior mistakes.",
      complexity: "High",
      notes: [
        "Builds trees sequentially rather than independently.",
        "Each new tree focuses on the residual errors left by the current ensemble.",
        "The final boundary is an additive score from many small corrections.",
      ],
      useCase: "Competitive tabular classification, nonlinear feature interactions, imbalanced datasets, and cases where predictive performance matters more than simple rules.",
      wiki: "https://en.wikipedia.org/wiki/XGBoost",
    },
    {
      id: "bayes",
      name: "Gaussian naive Bayes",
      subtitle: "Class likelihoods from simple bell-curve assumptions.",
      complexity: "Low",
      notes: [
        "Models each class as independent feature distributions.",
        "Classifies by the most likely class under those assumptions.",
        "Can work surprisingly well even when independence is imperfect.",
      ],
      useCase: "Fast baselines, small datasets, and text or event features where simple conditional evidence is useful.",
      wiki: "https://en.wikipedia.org/wiki/Naive_Bayes_classifier",
    },
  ],
  unsupervised: [
    {
      id: "kmeans",
      name: "K-means",
      subtitle: "Clusters points around movable centroids.",
      complexity: "Medium",
      notes: [
        "Alternates between assigning points and moving centroids.",
        "Prefers round clusters of similar size.",
        "You must choose k before the model starts.",
      ],
      useCase: "Segmentation, compression, summarizing similar records, and quick exploratory grouping.",
      wiki: "https://en.wikipedia.org/wiki/K-means_clustering",
    },
    {
      id: "dbscan",
      name: "DBSCAN",
      subtitle: "Dense regions become clusters; sparse points become noise.",
      complexity: "Medium",
      notes: [
        "Connects points that have enough neighbors within a radius.",
        "Finds irregular cluster shapes without selecting k.",
        "Very sensitive to the radius and density threshold.",
      ],
      useCase: "Geospatial clusters, anomaly detection, and data where noise should stay unassigned.",
      wiki: "https://en.wikipedia.org/wiki/DBSCAN",
    },
    {
      id: "spectral",
      name: "Spectral clustering",
      subtitle: "A local similarity graph separates shapes that centroids miss.",
      complexity: "High",
      notes: [
        "Builds a graph of nearby similar points.",
        "Groups points through graph connectivity rather than round distance-to-center regions.",
        "Works well on chains, moons, and rings where k-means tends to cut across the shape.",
      ],
      useCase: "Non-convex clusters, manifold-like data, and exploratory grouping where local neighborhood structure matters.",
      wiki: "https://en.wikipedia.org/wiki/Spectral_clustering",
    },
    {
      id: "optics",
      name: "OPTICS / HDBSCAN-style density",
      subtitle: "Clusters emerge from stable dense regions across neighborhood scales.",
      complexity: "High",
      notes: [
        "Uses core distances to compare density around each point.",
        "Connects points by mutual reachability rather than one fixed raw distance.",
        "Keeps sparse border points as noise when density is not stable enough.",
      ],
      useCase: "Variable-density clusters, anomaly-heavy data, and problems where plain DBSCAN is too sensitive to one radius.",
      wiki: "https://en.wikipedia.org/wiki/OPTICS_algorithm",
    },
    {
      id: "gmm",
      name: "Gaussian mixture",
      subtitle: "Soft clusters modeled as overlapping ellipses.",
      complexity: "High",
      notes: [
        "Estimates a probability of belonging to each cluster.",
        "Allows clusters to overlap instead of forcing hard borders immediately.",
        "Works best when groups look roughly elliptical.",
      ],
      useCase: "Probabilistic segmentation, uncertainty-aware grouping, and data with overlapping subpopulations.",
      wiki: "https://en.wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model",
    },
    {
      id: "hierarchical",
      name: "Agglomerative clustering",
      subtitle: "Small groups merge into larger groups step by step.",
      complexity: "High",
      notes: [
        "Starts with each point as its own group.",
        "Repeatedly merges the nearest groups.",
        "The final answer depends on where you cut the merge tree.",
      ],
      useCase: "Taxonomies, exploratory analysis, and cases where the nested structure is as important as the final groups.",
      wiki: "https://en.wikipedia.org/wiki/Hierarchical_clustering",
    },
  ],
};

const datasetCatalog = {
  supervised: [
    { id: "blobs", name: "Separated blobs" },
    { id: "overlap", name: "Overlapping blobs" },
    { id: "imbalanced", name: "Imbalanced classes" },
    { id: "anisotropic", name: "Anisotropic blobs" },
    { id: "xor", name: "XOR quadrants" },
    { id: "moons", name: "Interlocking moons" },
    { id: "rings", name: "Nested rings" },
    { id: "spiral", name: "Twin spirals" },
  ],
  unsupervised: [
    { id: "bridge", name: "Bridge / noisy connector" },
    { id: "clusters", name: "Cluster islands" },
    { id: "concentric-blobs", name: "Concentric blobs" },
    { id: "variable-size", name: "Variable size clusters" },
    { id: "density", name: "Variable density" },
    { id: "chain", name: "Curved chain" },
    { id: "elongated", name: "Elongated clusters" },
    { id: "nested-rings", name: "Nested rings" },
    { id: "outliers", name: "Outlier-heavy clusters" },
  ],
};

function byName(a, b) {
  return a.name.localeCompare(b.name);
}

function sortedByName(items) {
  return [...items].sort(byName);
}

  window.ClassificationVizShared = {
  palette: sharedPalette,
  noiseColor: sharedNoiseColor,
  modelCatalog,
  datasetCatalog,
  byName,
  sortedByName,
};

})();
