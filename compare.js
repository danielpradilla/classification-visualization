const WIDTH = 320;
const HEIGHT = 234;
const PAD = 24;
const PLOT = {
  left: PAD,
  right: WIDTH - PAD,
  top: PAD,
  bottom: HEIGHT - PAD,
};

const { palette, noiseColor, modelCatalog: sharedModelCatalog, datasetCatalog, sortedByName } = window.ClassificationVizShared;
const xScale = d3.scaleLinear().domain([-3, 3]).range([PLOT.left, PLOT.right]);
const yScale = d3.scaleLinear().domain([-3, 3]).range([PLOT.bottom, PLOT.top]);

const wallModelConfig = {
  logistic: {
    params: (p) => `C ${p.regularization.toFixed(1)}`,
    controls: [{ key: "regularization", label: "C", min: 0.1, max: 5, step: 0.1, digits: 1 }],
  },
  knn: {
    params: (p) => `k ${p.k}`,
    controls: [{ key: "k", label: "k", min: 2, max: 12, step: 1, digits: 0 }],
  },
  svm: {
    params: (p) => `C ${p.regularization.toFixed(1)}`,
    controls: [{ key: "regularization", label: "C", min: 0.1, max: 5, step: 0.1, digits: 1 }],
  },
  "rbf-svm": {
    params: (p) => `width ${p.kernelWidth.toFixed(2)}, C ${p.regularization.toFixed(1)}`,
    controls: [
      { key: "kernelWidth", label: "Width", min: 0.25, max: 1.8, step: 0.05, digits: 2 },
      { key: "regularization", label: "C", min: 0.1, max: 5, step: 0.1, digits: 1 },
    ],
  },
  "neural-net": {
    params: (p) => `${p.hiddenUnits} hidden, ${p.nnEpochs} epochs`,
    controls: [
      { key: "hiddenUnits", label: "Hidden", min: 3, max: 16, step: 1, digits: 0 },
      { key: "nnEpochs", label: "Epochs", min: 40, max: 320, step: 20, digits: 0 },
      { key: "nnRate", label: "Rate", min: 0.005, max: 0.08, step: 0.005, digits: 3 },
    ],
  },
  lda: {
    params: (p) => `shared covariance, smoothing ${p.smoothing.toFixed(2)}`,
    controls: [{ key: "smoothing", label: "Smooth", min: 0.01, max: 0.3, step: 0.01, digits: 2 }],
  },
  qda: {
    params: (p) => `separate covariance, smoothing ${p.smoothing.toFixed(2)}`,
    controls: [{ key: "smoothing", label: "Smooth", min: 0.01, max: 0.3, step: 0.01, digits: 2 }],
  },
  tree: {
    params: (p) => `depth ${p.depth}`,
    controls: [{ key: "depth", label: "Depth", min: 1, max: 6, step: 1, digits: 0 }],
  },
  forest: {
    params: (p) => `${p.forestTrees} trees, depth ${p.depth}`,
    controls: [
      { key: "forestTrees", label: "Trees", min: 3, max: 31, step: 2, digits: 0 },
      { key: "depth", label: "Depth", min: 1, max: 6, step: 1, digits: 0 },
    ],
  },
  xgboost: {
    params: (p) => `${p.boostRounds} rounds, rate ${p.boostRate.toFixed(2)}`,
    controls: [
      { key: "boostRounds", label: "Rounds", min: 2, max: 24, step: 1, digits: 0 },
      { key: "boostRate", label: "Rate", min: 0.1, max: 1, step: 0.04, digits: 2 },
      { key: "depth", label: "Depth", min: 1, max: 4, step: 1, digits: 0 },
    ],
  },
  bayes: {
    params: (p) => `independent features, smoothing ${p.smoothing.toFixed(2)}`,
    controls: [{ key: "smoothing", label: "Smooth", min: 0.01, max: 0.3, step: 0.01, digits: 2 }],
  },
  kmeans: {
    params: (p) => `${p.k} clusters`,
    controls: [{ key: "k", label: "Clusters", min: 2, max: 12, step: 1, digits: 0 }],
  },
  dbscan: {
    params: (p) => `radius ${p.radius.toFixed(2)}`,
    controls: [{ key: "radius", label: "Radius", min: 0.12, max: 0.52, step: 0.02, digits: 2 }],
  },
  spectral: {
    params: (p) => `${p.k} clusters, ${p.graphNeighbors} neighbors`,
    controls: [
      { key: "k", label: "Clusters", min: 2, max: 12, step: 1, digits: 0 },
      { key: "graphNeighbors", label: "Neighbors", min: 4, max: 20, step: 1, digits: 0 },
    ],
  },
  optics: {
    params: (p) => `${p.k} min pts, min size ${p.minClusterSize}`,
    controls: [
      { key: "k", label: "Min pts", min: 3, max: 10, step: 1, digits: 0 },
      { key: "radius", label: "Scale", min: 0.12, max: 0.52, step: 0.02, digits: 2 },
      { key: "minClusterSize", label: "Min size", min: 4, max: 40, step: 2, digits: 0 },
    ],
  },
  gmm: {
    params: (p) => `${p.k} components, ${p.spreadMode}`,
    controls: [
      { key: "k", label: "Components", min: 2, max: 12, step: 1, digits: 0 },
      { key: "spreadScale", label: "Spread", min: 0.65, max: 1.65, step: 0.05, digits: 2 },
    ],
  },
  hierarchical: {
    params: (p) => `${p.k} final groups`,
    controls: [{ key: "k", label: "Groups", min: 2, max: 12, step: 1, digits: 0 }],
  },
};

function modelCatalogForMode(mode) {
  return sharedModelCatalog[mode].map((model) => ({ ...model, ...wallModelConfig[model.id] }));
}

const state = {
  mode: "supervised",
  dataset: "moons",
  samples: 180,
  noise: 0.12,
  seed: 7,
  data: [],
  modelParams: {
    logistic: { regularization: 1 },
    knn: { k: 5 },
    svm: { regularization: 1 },
    "rbf-svm": { kernelWidth: 0.65, regularization: 1 },
    "neural-net": { hiddenUnits: 9, nnRate: 0.035, nnEpochs: 180 },
    lda: { smoothing: 0.08 },
    qda: { smoothing: 0.08 },
    tree: { depth: 3 },
    forest: { forestTrees: 9, depth: 3 },
    xgboost: { boostRounds: 6, boostRate: 0.62, depth: 2 },
    bayes: { smoothing: 0.1 },
    kmeans: { k: 4 },
    dbscan: { radius: 0.28 },
    spectral: { k: 3, graphNeighbors: 8 },
    optics: { k: 5, radius: 0.28, minClusterSize: 8 },
    gmm: { k: 4, spreadScale: 1, spreadMode: "elliptical" },
    hierarchical: { k: 4 },
  },
};

let activeParams = {};
const cardsByModel = new Map();

const els = {
  datasetSelect: document.querySelector("#datasetSelect"),
  sampleSlider: document.querySelector("#sampleSlider"),
  noiseSlider: document.querySelector("#noiseSlider"),
  sampleValue: document.querySelector("#sampleValue"),
  noiseValue: document.querySelector("#noiseValue"),
  regenerateButton: document.querySelector("#regenerateButton"),
  modelWall: document.querySelector("#modelWall"),
  bestModel: document.querySelector("#bestModel"),
  readSummary: document.querySelector("#readSummary"),
  scoreSummaryLabel: document.querySelector("#scoreSummaryLabel"),
};

function param(key, fallback) {
  return activeParams[key] ?? fallback;
}

function currentDatasets() {
  return sortedByName(datasetCatalog[state.mode]);
}

function currentModels() {
  return sortedByName(modelCatalogForMode(state.mode));
}

function rng(seed) {
  let value = seed % 2147483647;
  return () => {
    value = (value * 16807) % 2147483647;
    return (value - 1) / 2147483646;
  };
}

function gaussian(random) {
  const u = Math.max(random(), 1e-9);
  const v = Math.max(random(), 1e-9);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function dist2(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return dx * dx + dy * dy;
}

function generateData() {
  const random = rng(state.seed);
  const data = [];
  const n = state.samples;
  const noise = state.noise;

  for (let i = 0; i < n; i += 1) {
    let x;
    let y;
    let label = 0;

    if (state.mode === "supervised" && state.dataset === "xor") {
      x = -2.3 + random() * 4.6;
      y = -2.3 + random() * 4.6;
      label = x * y > 0 ? 0 : 1;
      x += gaussian(random) * noise;
      y += gaussian(random) * noise;
    } else if (state.mode === "supervised" && state.dataset === "spiral") {
      const second = i >= n / 2;
      const t = 0.7 + random() * 3.9;
      const r = 0.48 * t;
      const angle = second ? t + Math.PI : t;
      x = Math.cos(angle) * r + gaussian(random) * (0.08 + noise / 1.5);
      y = Math.sin(angle) * r + gaussian(random) * (0.08 + noise / 1.5);
      label = second ? 1 : 0;
    } else if (state.mode === "supervised" && state.dataset === "moons") {
      const upper = i < n / 2;
      const t = random() * Math.PI;
      x = upper ? Math.cos(t) * 1.8 - 0.35 : 1 - Math.cos(t) * 1.8 + 0.35;
      y = upper ? Math.sin(t) * 1.15 - 0.35 : -Math.sin(t) * 1.15 + 0.45;
      label = upper ? 0 : 1;
      x += gaussian(random) * (0.18 + noise);
      y += gaussian(random) * (0.18 + noise);
    } else if (state.mode === "supervised" && state.dataset === "rings") {
      const inner = i < n / 2;
      const r = inner ? 0.9 + gaussian(random) * (0.07 + noise / 2) : 2.05 + gaussian(random) * (0.08 + noise / 2);
      const t = random() * Math.PI * 2;
      x = Math.cos(t) * r + gaussian(random) * noise;
      y = Math.sin(t) * r + gaussian(random) * noise;
      label = inner ? 0 : 1;
    } else if (state.mode === "supervised" && state.dataset === "imbalanced") {
      const minority = i >= Math.floor(n * 0.82);
      x = (minority ? 1.2 : -0.75) + gaussian(random) * (minority ? 0.42 + noise : 0.72 + noise);
      y = (minority ? 0.85 : -0.25) + gaussian(random) * (minority ? 0.42 + noise : 0.62 + noise);
      label = minority ? 1 : 0;
    } else if (state.mode === "supervised" && state.dataset === "overlap") {
      const right = i < n / 2;
      x = (right ? 0.72 : -0.72) + gaussian(random) * (0.88 + noise * 1.7);
      y = (right ? 0.28 : -0.28) + gaussian(random) * (0.82 + noise * 1.4);
      label = right ? 1 : 0;
    } else if (state.mode === "supervised" && state.dataset === "anisotropic") {
      const right = i < n / 2;
      const u = gaussian(random);
      const v = gaussian(random);
      const angle = right ? -0.62 : 0.72;
      const long = 1.15 + noise * 1.6;
      const short = 0.28 + noise;
      const cx = right ? 1.08 : -1.08;
      const cy = right ? 0.28 : -0.28;
      x = cx + Math.cos(angle) * u * long - Math.sin(angle) * v * short;
      y = cy + Math.sin(angle) * u * long + Math.cos(angle) * v * short;
      label = right ? 1 : 0;
    } else if (state.mode === "supervised") {
      const left = i < n / 2;
      x = (left ? -1.15 : 1.15) + gaussian(random) * (0.62 + noise);
      y = (left ? -0.45 : 0.45) + gaussian(random) * (0.56 + noise);
      label = left ? 0 : 1;
    } else if (state.dataset === "nested-rings") {
      const ring = i % 3;
      const radii = [0.62, 1.38, 2.18];
      const t = random() * Math.PI * 2;
      const r = radii[ring] + gaussian(random) * (0.06 + noise / 2);
      x = Math.cos(t) * r + gaussian(random) * noise * 0.45;
      y = Math.sin(t) * r + gaussian(random) * noise * 0.45;
      label = ring;
    } else if (state.dataset === "variable-size") {
      const bucket = random();
      const cluster = bucket < 0.52 ? 0 : bucket < 0.76 ? 1 : bucket < 0.92 ? 2 : 3;
      const centers = [
        { x: -1.55, y: -0.95, s: 0.78 },
        { x: 1.35, y: 1.08, s: 0.28 },
        { x: -1.25, y: 1.3, s: 0.42 },
        { x: 1.6, y: -1.2, s: 0.2 },
      ];
      const c = centers[cluster];
      x = c.x + gaussian(random) * (c.s + noise);
      y = c.y + gaussian(random) * (c.s * 0.84 + noise);
      label = cluster;
    } else if (state.dataset === "density") {
      const cluster = i % 3;
      const centers = [
        { x: -1.45, y: 0.9, s: 0.32 },
        { x: 1.25, y: 0.72, s: 0.7 },
        { x: 0.2, y: -1.25, s: 0.46 },
      ];
      const c = centers[cluster];
      x = c.x + gaussian(random) * (c.s + noise);
      y = c.y + gaussian(random) * (c.s * 0.72 + noise);
      label = cluster;
    } else if (state.dataset === "chain") {
      const t = -2.5 + random() * 5;
      x = t + gaussian(random) * (0.11 + noise / 1.7);
      y = Math.sin(t * 1.65) * 1.05 + gaussian(random) * (0.17 + noise / 1.4);
      label = t < -0.9 ? 0 : t < 0.9 ? 1 : 2;
    } else {
      const cluster = i % 4;
      const centers = [
        { x: -1.5, y: -1.05 },
        { x: -1.1, y: 1.15 },
        { x: 1.25, y: 0.98 },
        { x: 1.45, y: -1.15 },
      ];
      const c = centers[cluster];
      x = c.x + gaussian(random) * (0.34 + noise);
      y = c.y + gaussian(random) * (0.34 + noise);
      label = cluster;
    }

    data.push({
      id: i,
      x: clamp(x, -2.9, 2.9),
      y: clamp(y, -2.9, 2.9),
      label,
      isTest: state.mode === "supervised" && random() > 0.78,
    });
  }

  state.data = data;
}

function trainModel(meta) {
  activeParams = state.modelParams[meta.id] || {};
  const train = state.data.filter((d) => !d.isTest);

  if (state.mode === "unsupervised") {
    if (meta.id === "dbscan") return trainDbscan(state.data);
    if (meta.id === "spectral") return trainSpectral(state.data);
    if (meta.id === "optics") return trainOptics(state.data);
    if (meta.id === "gmm") return trainGmm(state.data);
    if (meta.id === "hierarchical") return trainHierarchical(state.data);
    return trainKmeans(state.data);
  }

  if (meta.id === "knn") return trainKnn(train);
  if (meta.id === "svm") return trainLinear(train, "svm");
  if (meta.id === "rbf-svm") return trainRbfSvm(train);
  if (meta.id === "neural-net") return trainNeuralNet(train);
  if (meta.id === "lda") return trainDiscriminant(train, "lda");
  if (meta.id === "qda") return trainDiscriminant(train, "qda");
  if (meta.id === "tree") return trainTree(train, param("depth", 3));
  if (meta.id === "forest") return trainForest(train, param("depth", 3));
  if (meta.id === "xgboost") return trainBoostedTrees(train);
  if (meta.id === "bayes") return trainBayes(train);
  return trainLinear(train, "logistic");
}

function trainLinear(train, kind) {
  let w0 = 0;
  let w1 = 0;
  let b = 0;
  const rate = kind === "svm" ? 0.012 : 0.055;
  const c = param("regularization", 1);
  const shrink = 0.01 / Math.max(0.1, c);

  for (let iter = 0; iter < 220; iter += 1) {
    for (const p of train) {
      const y = p.label === 1 ? 1 : 0;
      const score = w0 * p.x + w1 * p.y + b;
      if (kind === "svm") {
        const target = y === 1 ? 1 : -1;
        if (target * score < 1) {
          w0 += rate * (c * target * p.x - shrink * w0);
          w1 += rate * (c * target * p.y - shrink * w1);
          b += rate * c * target;
        } else {
          w0 -= rate * shrink * w0;
          w1 -= rate * shrink * w1;
        }
      } else {
        const pred = 1 / (1 + Math.exp(-score));
        const error = y - pred;
        w0 += rate * (error * p.x - shrink * w0);
        w1 += rate * (error * p.y - shrink * w1);
        b += rate * error;
      }
    }
  }

  return {
    type: "linear",
    w0,
    w1,
    b,
    predict: (p) => (w0 * p.x + w1 * p.y + b >= 0 ? 1 : 0),
  };
}

function trainKnn(train) {
  const k = Math.min(param("k", 5), train.length);
  return {
    type: "knn",
    predict: (p) => {
      const votes = new Map();
      train
        .map((d) => ({ d, distance: dist2(d, p) }))
        .sort((a, b) => a.distance - b.distance)
        .slice(0, k)
        .forEach(({ d }) => votes.set(d.label, (votes.get(d.label) || 0) + 1));
      return [...votes.entries()].sort((a, b) => b[1] - a[1])[0][0];
    },
  };
}

function trainRbfSvm(train) {
  const gamma = 1 / (2 * param("kernelWidth", 0.65) ** 2);
  const c = param("regularization", 1);
  const alpha = new Array(train.length).fill(0);
  const targets = train.map((d) => (d.label === 1 ? 1 : -1));
  const kernel = (a, b) => Math.exp(-gamma * dist2(a, b));

  for (let epoch = 0; epoch < 6; epoch += 1) {
    for (let i = 0; i < train.length; i += 1) {
      let score = 0;
      for (let j = 0; j < train.length; j += 1) {
        if (alpha[j] === 0) continue;
        score += alpha[j] * targets[j] * kernel(train[j], train[i]);
      }
      if (targets[i] * score <= 0.15) alpha[i] = Math.min(c * 4, alpha[i] + 0.6 * c);
    }
  }

  const supportVectors = train.map((d, i) => ({ ...d, alpha: alpha[i], target: targets[i] })).filter((d) => d.alpha > 0);
  return {
    type: "kernel",
    supportVectors,
    predict: (p) => {
      const score = supportVectors.reduce((sum, d) => sum + d.alpha * d.target * kernel(d, p), 0);
      return score >= 0 ? 1 : 0;
    },
  };
}

function trainNeuralNet(train) {
  const random = rng(state.seed + 501);
  const hiddenCount = param("hiddenUnits", 9);
  const hidden = d3.range(hiddenCount).map(() => ({
    wx: gaussian(random) * 0.55,
    wy: gaussian(random) * 0.55,
    b: gaussian(random) * 0.12,
  }));
  const output = d3.range(hiddenCount).map(() => gaussian(random) * 0.45);
  let outputBias = 0;
  const rate = param("nnRate", 0.035);

  for (let iter = 0; iter < param("nnEpochs", 180); iter += 1) {
    for (const p of train) {
      const activations = hidden.map((h) => Math.tanh(h.wx * p.x + h.wy * p.y + h.b));
      const score = outputBias + d3.sum(activations, (a, i) => a * output[i]);
      const pred = 1 / (1 + Math.exp(-score));
      const error = pred - p.label;
      outputBias -= rate * error;
      for (let i = 0; i < hiddenCount; i += 1) {
        const oldOut = output[i];
        output[i] -= rate * error * activations[i];
        const hiddenGrad = error * oldOut * (1 - activations[i] ** 2);
        hidden[i].wx -= rate * hiddenGrad * p.x;
        hidden[i].wy -= rate * hiddenGrad * p.y;
        hidden[i].b -= rate * hiddenGrad;
      }
    }
  }

  const scorePoint = (p) => outputBias + d3.sum(hidden, (h, i) => Math.tanh(h.wx * p.x + h.wy * p.y + h.b) * output[i]);
  return {
    type: "neural",
    hidden,
    predict: (p) => (scorePoint(p) >= 0 ? 1 : 0),
  };
}

function covarianceStats(rows, smoothing = 0.08) {
  const meanX = d3.mean(rows, (d) => d.x) || 0;
  const meanY = d3.mean(rows, (d) => d.y) || 0;
  const denom = Math.max(1, rows.length - 1);
  const xx = d3.sum(rows, (d) => (d.x - meanX) ** 2) / denom + smoothing;
  const yy = d3.sum(rows, (d) => (d.y - meanY) ** 2) / denom + smoothing;
  const xy = d3.sum(rows, (d) => (d.x - meanX) * (d.y - meanY)) / denom;
  const det = Math.max(0.02, xx * yy - xy * xy);
  return {
    meanX,
    meanY,
    xx,
    yy,
    xy,
    det,
    inv00: yy / det,
    inv01: -xy / det,
    inv11: xx / det,
  };
}

function trainDiscriminant(train, kind) {
  const smoothing = param("smoothing", 0.08);
  const groups = d3.group(train, (d) => d.label);
  const classStats = [...groups.entries()].map(([label, rows]) => ({
    label,
    prior: rows.length / train.length,
    covariance: covarianceStats(rows, smoothing),
    rows,
  }));

  if (kind === "lda") {
    const pooled = covarianceStats(train, smoothing);
    classStats.forEach((stat) => {
      const classCovariance = covarianceStats(stat.rows, smoothing);
      stat.covariance = {
        ...pooled,
        meanX: classCovariance.meanX,
        meanY: classCovariance.meanY,
      };
    });
  }

  const score = (p, stat) => {
    const c = stat.covariance;
    const dx = p.x - c.meanX;
    const dy = p.y - c.meanY;
    const mahalanobis = dx * (c.inv00 * dx + c.inv01 * dy) + dy * (c.inv01 * dx + c.inv11 * dy);
    const logDet = kind === "lda" ? 0 : Math.log(c.det);
    return Math.log(stat.prior) - 0.5 * logDet - 0.5 * mahalanobis;
  };

  return {
    type: kind,
    classStats,
    predict: (p) => classStats.map((stat) => ({ label: stat.label, score: score(p, stat) })).sort((a, b) => b.score - a.score)[0].label,
  };
}

function gini(rows) {
  if (!rows.length) return 0;
  const counts = d3.rollup(rows, (v) => v.length, (d) => d.label);
  let impurity = 1;
  for (const count of counts.values()) {
    const p = count / rows.length;
    impurity -= p * p;
  }
  return impurity;
}

function majority(rows) {
  const counts = d3.rollup(rows, (v) => v.length, (d) => d.label);
  return [...counts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] ?? 0;
}

function buildTree(rows, depth, bounds, featureMask = ["x", "y"]) {
  if (depth === 0 || gini(rows) < 0.02 || rows.length < 8) {
    return { leaf: true, label: majority(rows), bounds };
  }

  let best = null;
  for (const feature of featureMask) {
    const values = rows.map((d) => d[feature]).sort((a, b) => a - b);
    const candidates = d3.range(6, values.length - 6, Math.max(4, Math.floor(values.length / 18))).map((i) => values[i]);
    for (const threshold of candidates) {
      const left = rows.filter((d) => d[feature] <= threshold);
      const right = rows.filter((d) => d[feature] > threshold);
      const score = (left.length * gini(left) + right.length * gini(right)) / rows.length;
      if (!best || score < best.score) best = { feature, threshold, left, right, score };
    }
  }

  if (!best) return { leaf: true, label: majority(rows), bounds };
  const leftBounds = { ...bounds };
  const rightBounds = { ...bounds };
  if (best.feature === "x") {
    leftBounds.xMax = best.threshold;
    rightBounds.xMin = best.threshold;
  } else {
    leftBounds.yMax = best.threshold;
    rightBounds.yMin = best.threshold;
  }

  return {
    leaf: false,
    label: majority(rows),
    feature: best.feature,
    threshold: best.threshold,
    bounds,
    left: buildTree(best.left, depth - 1, leftBounds, featureMask),
    right: buildTree(best.right, depth - 1, rightBounds, featureMask),
  };
}

function predictTree(node, p) {
  if (node.leaf) return node.label;
  return predictTree(p[node.feature] <= node.threshold ? node.left : node.right, p);
}

function collectSplits(node, list = []) {
  if (node.leaf) return list;
  list.push(node);
  collectSplits(node.left, list);
  collectSplits(node.right, list);
  return list;
}

function trainTree(train, depth) {
  const tree = buildTree(train, depth, { xMin: -3, xMax: 3, yMin: -3, yMax: 3 });
  return {
    type: "tree",
    tree,
    splits: collectSplits(tree),
    predict: (p) => predictTree(tree, p),
  };
}

function trainForest(train, depth) {
  const random = rng(state.seed + 99);
  const treeCount = param("forestTrees", 9);
  const trees = d3.range(treeCount).map((i) => {
    const sample = d3.range(train.length).map(() => train[Math.floor(random() * train.length)]);
    const features = i % 3 === 0 ? ["x"] : i % 3 === 1 ? ["y"] : ["x", "y"];
    return buildTree(sample, Math.max(2, depth - (i % 2)), { xMin: -3, xMax: 3, yMin: -3, yMax: 3 }, features);
  });
  return {
    type: "forest",
    trees,
    predict: (p) => majority(trees.map((tree) => ({ label: predictTree(tree, p) }))),
  };
}

function squaredError(rows) {
  if (!rows.length) return 0;
  const mean = d3.mean(rows, (d) => d.residual) || 0;
  return d3.sum(rows, (d) => (d.residual - mean) ** 2);
}

function buildRegressionTree(rows, depth, bounds) {
  const value = d3.mean(rows, (d) => d.residual) || 0;
  if (depth === 0 || rows.length < 10 || squaredError(rows) < 0.001) {
    return { leaf: true, value, bounds };
  }

  let best = null;
  for (const feature of ["x", "y"]) {
    const values = rows.map((d) => d[feature]).sort((a, b) => a - b);
    const stride = Math.max(5, Math.floor(values.length / 16));
    const candidates = d3.range(stride, values.length - stride, stride).map((i) => values[i]);
    for (const threshold of candidates) {
      const left = rows.filter((d) => d[feature] <= threshold);
      const right = rows.filter((d) => d[feature] > threshold);
      const score = squaredError(left) + squaredError(right);
      if (!best || score < best.score) best = { feature, threshold, left, right, score };
    }
  }

  if (!best) return { leaf: true, value, bounds };
  const leftBounds = { ...bounds };
  const rightBounds = { ...bounds };
  if (best.feature === "x") {
    leftBounds.xMax = best.threshold;
    rightBounds.xMin = best.threshold;
  } else {
    leftBounds.yMax = best.threshold;
    rightBounds.yMin = best.threshold;
  }

  return {
    leaf: false,
    value,
    feature: best.feature,
    threshold: best.threshold,
    bounds,
    left: buildRegressionTree(best.left, depth - 1, leftBounds),
    right: buildRegressionTree(best.right, depth - 1, rightBounds),
  };
}

function predictRegressionTree(node, p) {
  if (node.leaf) return node.value;
  return predictRegressionTree(p[node.feature] <= node.threshold ? node.left : node.right, p);
}

function trainBoostedTrees(train) {
  const learningRate = param("boostRate", 0.62);
  const positiveRate = clamp(d3.mean(train, (d) => d.label) || 0.5, 0.05, 0.95);
  const initialScore = Math.log(positiveRate / (1 - positiveRate));
  const scores = new Map(train.map((d) => [d.id, initialScore]));
  const trees = [];
  const rounds = param("boostRounds", 6);
  const depth = Math.min(4, Math.max(1, param("depth", 2)));

  for (let round = 0; round < rounds; round += 1) {
    const residualRows = train.map((d) => {
      const probability = 1 / (1 + Math.exp(-(scores.get(d.id) || 0)));
      return { ...d, residual: d.label - probability };
    });
    const tree = buildRegressionTree(residualRows, depth, { xMin: -3, xMax: 3, yMin: -3, yMax: 3 });
    trees.push(tree);
    for (const d of train) {
      scores.set(d.id, (scores.get(d.id) || 0) + learningRate * predictRegressionTree(tree, d));
    }
  }

  const splits = trees.flatMap((tree, learner) => collectSplits(tree).map((split) => ({ ...split, learner })));

  return {
    type: "boosted",
    trees,
    splits,
    predict: (p) => {
      const score = trees.reduce((acc, tree) => acc + learningRate * predictRegressionTree(tree, p), initialScore);
      return score >= 0 ? 1 : 0;
    },
  };
}

function trainBayes(train) {
  const smoothing = param("smoothing", 0.1);
  const groups = d3.group(train, (d) => d.label);
  const stats = [...groups.entries()].map(([label, rows]) => {
    const meanX = d3.mean(rows, (d) => d.x);
    const meanY = d3.mean(rows, (d) => d.y);
    const varX = (d3.variance(rows, (d) => d.x) || 0.1) + smoothing;
    const varY = (d3.variance(rows, (d) => d.y) || 0.1) + smoothing;
    return { label, prior: rows.length / train.length, meanX, meanY, varX, varY };
  });
  const logPdf = (value, mean, variance) => -0.5 * Math.log(2 * Math.PI * variance) - ((value - mean) ** 2) / (2 * variance);
  return {
    type: "bayes",
    stats,
    predict: (p) =>
      stats
        .map((s) => ({ label: s.label, score: Math.log(s.prior) + logPdf(p.x, s.meanX, s.varX) + logPdf(p.y, s.meanY, s.varY) }))
        .sort((a, b) => b.score - a.score)[0].label,
  };
}

function nearestCentroid(p, centroids) {
  return centroids
    .map((c, i) => ({ i, distance: dist2(p, c) }))
    .sort((a, b) => a.distance - b.distance)[0].i;
}

function trainKmeans(data) {
  const k = clamp(param("k", 4), 2, 12);
  let centroids = d3.range(k).map((i) => {
    const angle = (i / k) * Math.PI * 2;
    const ring = i < 8 ? 1.35 : 2.15;
    return { x: Math.cos(angle) * ring, y: Math.sin(angle) * ring, label: i };
  });
  let assignments = [];

  for (let iter = 0; iter < 8; iter += 1) {
    assignments = data.map((p) => nearestCentroid(p, centroids));
    centroids = centroids.map((c, i) => {
      const members = data.filter((_, idx) => assignments[idx] === i);
      return members.length ? { x: d3.mean(members, (d) => d.x), y: d3.mean(members, (d) => d.y), label: i } : c;
    });
  }

  return {
    type: "centroid",
    centroids,
    assignments,
    predict: (p) => nearestCentroid(p, centroids),
  };
}

function trainDbscan(data) {
  const radius = param("radius", 0.28);
  const minPts = 5;
  const labels = new Array(data.length).fill(undefined);
  let cluster = 0;
  const neighbors = (idx) => data.map((p, i) => ({ p, i })).filter(({ p }) => Math.sqrt(dist2(data[idx], p)) <= radius).map(({ i }) => i);

  for (let i = 0; i < data.length; i += 1) {
    if (labels[i] !== undefined) continue;
    const queue = neighbors(i);
    if (queue.length < minPts) {
      labels[i] = -1;
      continue;
    }
    labels[i] = cluster;
    for (let q = 0; q < queue.length; q += 1) {
      const j = queue[q];
      if (labels[j] === -1) labels[j] = cluster;
      if (labels[j] !== undefined) continue;
      labels[j] = cluster;
      const more = neighbors(j);
      if (more.length >= minPts) queue.push(...more.filter((m) => !queue.includes(m)));
    }
    cluster += 1;
  }

  return {
    type: "density",
    labels,
    clusters: cluster,
    radius,
    predict: (p) => {
      const close = data
        .map((d, i) => ({ i, distance: dist2(d, p) }))
        .sort((a, b) => a.distance - b.distance)[0];
      return close && Math.sqrt(close.distance) <= radius * 1.5 ? labels[close.i] : -1;
    },
  };
}

function nearestPointLabel(p, data, labels, maxDistance = Infinity) {
  const close = data
    .map((d, i) => ({ i, distance: dist2(d, p) }))
    .sort((a, b) => a.distance - b.distance)[0];
  if (!close || Math.sqrt(close.distance) > maxDistance) return -1;
  return labels[close.i] ?? -1;
}

function componentLabelsFromEdges(data, edges, targetClusters, minSize = 1) {
  const parent = data.map((_, i) => i);
  const find = (i) => {
    while (parent[i] !== i) {
      parent[i] = parent[parent[i]];
      i = parent[i];
    }
    return i;
  };
  const union = (a, b) => {
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) parent[rb] = ra;
  };

  const sorted = [...edges].sort((a, b) => a.weight - b.weight);
  let components = data.length;
  for (const edge of sorted) {
    if (components <= targetClusters) break;
    const ra = find(edge.a);
    const rb = find(edge.b);
    if (ra !== rb) {
      union(ra, rb);
      components -= 1;
    }
  }

  const grouped = d3.group(data.map((d, i) => ({ d, i, root: find(i) })), (d) => d.root);
  const rootToLabel = new Map();
  let label = 0;
  for (const [root, rows] of grouped.entries()) {
    if (rows.length >= minSize) {
      rootToLabel.set(root, label);
      label += 1;
    }
  }
  return data.map((_, i) => rootToLabel.get(find(i)) ?? -1);
}

function trainSpectral(data) {
  const targetClusters = clamp(param("k", 3), 2, 12);
  const graphK = param("graphNeighbors", 8);
  const edges = [];
  for (let i = 0; i < data.length; i += 1) {
    data
      .map((d, j) => ({ j, distance: dist2(data[i], d) }))
      .filter((d) => d.j !== i)
      .sort((a, b) => a.distance - b.distance)
      .slice(0, graphK)
      .forEach(({ j, distance }) => {
        if (i < j) edges.push({ a: i, b: j, weight: Math.sqrt(distance) });
      });
  }
  const labels = componentLabelsFromEdges(data, edges, targetClusters);
  const centroids = [...d3.group(data.map((d, i) => ({ ...d, assigned: labels[i] })).filter((d) => d.assigned !== -1), (d) => d.assigned).entries()].map(
    ([label, rows]) => ({ label, x: d3.mean(rows, (d) => d.x), y: d3.mean(rows, (d) => d.y) }),
  );
  return {
    type: "graph",
    labels,
    centroids,
    links: edges.slice(0, 160),
    predict: (p) => nearestPointLabel(p, data, labels),
  };
}

function trainOptics(data) {
  const minPts = clamp(param("k", 5), 3, 10);
  const minClusterSize = param("minClusterSize", 8);
  const sortedDistances = data.map((p, i) =>
    data
      .map((d, j) => (i === j ? Infinity : Math.sqrt(dist2(p, d))))
      .sort((a, b) => a - b),
  );
  const coreDistance = sortedDistances.map((distances) => distances[minPts - 1] || Infinity);
  const finiteCoreDistances = coreDistance.filter(Number.isFinite).sort((a, b) => a - b);
  const adaptiveCut = (d3.quantile(finiteCoreDistances, 0.68) || param("radius", 0.28)) * 1.35;
  const reachabilityCut = Math.max(param("radius", 0.28) * 2.3, adaptiveCut);
  const edges = [];
  for (let i = 0; i < data.length; i += 1) {
    data
      .map((d, j) => ({ j, raw: Math.sqrt(dist2(data[i], d)) }))
      .filter((d) => d.j !== i)
      .sort((a, b) => a.raw - b.raw)
      .slice(0, minPts + 2)
      .forEach(({ j, raw }) => {
        if (i < j) edges.push({ a: i, b: j, weight: Math.max(coreDistance[i], coreDistance[j], raw) });
      });
  }
  const stableEdges = edges.filter((edge) => edge.weight <= reachabilityCut);
  const labels = componentLabelsFromEdges(data, stableEdges, 1, minClusterSize);
  const centroids = [...d3.group(data.map((d, i) => ({ ...d, assigned: labels[i] })).filter((d) => d.assigned !== -1), (d) => d.assigned).entries()].map(
    ([label, rows]) => ({ label, x: d3.mean(rows, (d) => d.x), y: d3.mean(rows, (d) => d.y) }),
  );
  return {
    type: "optics",
    labels,
    centroids,
    links: stableEdges.slice(0, 180),
    radius: param("radius", 0.28),
    predict: (p) => nearestPointLabel(p, data, labels, reachabilityCut * 1.25),
  };
}

function trainGmm(data) {
  const result = trainKmeans(data);
  const spreadScale = param("spreadScale", 1);
  const spreads = result.centroids.map((c, i) => {
    const members = data.filter((_, idx) => result.assignments[idx] === i);
    const sx = Math.max(0.18, Math.sqrt(d3.mean(members, (d) => (d.x - c.x) ** 2) || 0.18)) * spreadScale;
    const sy = Math.max(0.18, Math.sqrt(d3.mean(members, (d) => (d.y - c.y) ** 2) || 0.18)) * spreadScale;
    return { x: sx, y: sy };
  });
  const softResults = data.map((p) => softClusterResult(p, result.centroids, spreads));
  return {
    type: "gmm",
    centroids: result.centroids,
    spreads,
    confidences: softResults.map((d) => d.confidence),
    assignments: softResults.map((d) => d.label),
    predict: (p) => softClusterResult(p, result.centroids, spreads).label,
  };
}

function softClusterResult(p, centroids, spreads) {
  const scores = centroids
    .map((c, i) => {
      const sx = spreads[i].x || spreads[i];
      const sy = spreads[i].y || spreads[i];
      const dx = (p.x - c.x) / sx;
      const dy = (p.y - c.y) / sy;
      return { i, score: Math.exp(-(dx * dx + dy * dy) / 2) };
    })
    .sort((a, b) => b.score - a.score);
  const total = d3.sum(scores, (d) => d.score) || 1;
  return { label: scores[0].i, confidence: scores[0].score / total };
}

function trainHierarchical(data) {
  const k = clamp(param("k", 4), 2, 12);
  let groups = data.map((p, i) => ({ id: i, points: [p], x: p.x, y: p.y }));
  while (groups.length > k) {
    let best = { a: 0, b: 1, distance: Infinity };
    for (let i = 0; i < groups.length; i += 1) {
      for (let j = i + 1; j < groups.length; j += 1) {
        const distance = dist2(groups[i], groups[j]);
        if (distance < best.distance) best = { a: i, b: j, distance };
      }
    }
    const a = groups[best.a];
    const b = groups[best.b];
    const mergedPoints = [...a.points, ...b.points];
    const merged = {
      id: a.id,
      points: mergedPoints,
      x: d3.mean(mergedPoints, (d) => d.x),
      y: d3.mean(mergedPoints, (d) => d.y),
    };
    groups = groups.filter((_, idx) => idx !== best.a && idx !== best.b);
    groups.push(merged);
  }

  const centroids = groups.map((g, i) => ({ x: g.x, y: g.y, label: i }));
  return {
    type: "hierarchical",
    centroids,
    assignments: data.map((p) => nearestCentroid(p, centroids)),
    predict: (p) => nearestCentroid(p, centroids),
  };
}

function evaluate(model) {
  if (state.mode === "unsupervised") {
    const assignments = state.data.map((d, i) => model.assignments?.[i] ?? model.labels?.[i] ?? model.predict(d));
    const clustered = assignments.filter((v) => v !== -1);
    const unique = new Set(clustered);
    const noise = assignments.length - clustered.length;
    return {
      score: unique.size,
      label: `${unique.size} clusters${noise ? `, ${noise} noise` : ""}`,
      detail: `${clustered.length}/${assignments.length} assigned`,
      noise,
    };
  }

  const test = state.data.filter((d) => d.isTest);
  const evalSet = test.length ? test : state.data;
  const correct = evalSet.filter((d) => model.predict(d) === d.label).length;
  const accuracy = Math.round((correct / evalSet.length) * 100);
  return {
    score: accuracy,
    label: `${accuracy}%`,
    detail: `${correct}/${evalSet.length} holdout correct`,
    correct,
    total: evalSet.length,
  };
}

function drawAxes(svg) {
  svg
    .append("rect")
    .attr("class", "axis-frame")
    .attr("x", PLOT.left)
    .attr("y", PLOT.top)
    .attr("width", PLOT.right - PLOT.left)
    .attr("height", PLOT.bottom - PLOT.top);

  svg.append("line").attr("class", "axis-line").attr("x1", xScale(-3)).attr("x2", xScale(3)).attr("y1", yScale(0)).attr("y2", yScale(0));
  svg.append("line").attr("class", "axis-line").attr("x1", xScale(0)).attr("x2", xScale(0)).attr("y1", yScale(-3)).attr("y2", yScale(3));
}

function drawDecisionMap(svg, model) {
  const cells = [];
  const grid = 26;
  for (let ix = 0; ix < grid; ix += 1) {
    for (let iy = 0; iy < grid; iy += 1) {
      const x = -3 + (ix / grid) * 6;
      const y = -3 + (iy / grid) * 6;
      cells.push({ x, y, label: model.predict({ x, y }) });
    }
  }

  const cellW = (PLOT.right - PLOT.left) / grid + 1;
  const cellH = (PLOT.bottom - PLOT.top) / grid + 1;
  svg
    .append("g")
    .selectAll("rect")
    .data(cells)
    .join("rect")
    .attr("class", "domain-bg")
    .attr("x", (d) => xScale(d.x))
    .attr("y", (d) => yScale(d.y) - cellH)
    .attr("width", cellW)
    .attr("height", cellH)
    .attr("fill", (d) => (d.label === -1 ? noiseColor : palette[Math.abs(d.label) % palette.length]));
}

function drawOverlays(svg, model) {
  const layer = svg.append("g");
  if (model.type === "linear") {
    const points = [-3, 3].map((x) => ({ x, y: -(model.w0 * x + model.b) / (model.w1 || 0.0001) }));
    layer
      .append("path")
      .attr("class", "boundary-line")
      .attr("d", d3.line().x((d) => xScale(d.x)).y((d) => yScale(d.y))(points));
  }

  if (model.type === "kernel") {
    layer
      .selectAll("circle")
      .data(model.supportVectors.slice(0, 80))
      .join("circle")
      .attr("class", "support-vector")
      .attr("cx", (d) => xScale(d.x))
      .attr("cy", (d) => yScale(d.y))
      .attr("r", 5.8);
  }

  if (model.type === "tree" || model.type === "boosted") {
    layer
      .selectAll("line")
      .data(model.type === "boosted" ? model.splits.slice(0, 14) : model.splits)
      .join("line")
      .attr("class", "tree-split")
      .attr("x1", (d) => (d.feature === "x" ? xScale(d.threshold) : xScale(d.bounds.xMin)))
      .attr("x2", (d) => (d.feature === "x" ? xScale(d.threshold) : xScale(d.bounds.xMax)))
      .attr("y1", (d) => (d.feature === "x" ? yScale(d.bounds.yMin) : yScale(d.threshold)))
      .attr("y2", (d) => (d.feature === "x" ? yScale(d.bounds.yMax) : yScale(d.threshold)))
      .attr("opacity", (d) => (model.type === "boosted" ? Math.max(0.12, 0.62 - d.learner * 0.05) : 0.58));
  }

  if (["centroid", "gmm", "hierarchical", "graph", "optics"].includes(model.type)) {
    layer
      .selectAll("path.centroid")
      .data(model.centroids)
      .join("path")
      .attr("class", "centroid")
      .attr("d", d3.symbol().type(d3.symbolCross).size(90))
      .attr("transform", (d) => `translate(${xScale(d.x)},${yScale(d.y)})`)
      .attr("fill", (d, i) => palette[i % palette.length]);
  }

  if (model.type === "graph" || model.type === "optics") {
    layer
      .selectAll("line.graph-link")
      .data(model.links || [])
      .join("line")
      .attr("class", "graph-link")
      .attr("x1", (d) => xScale(state.data[d.a].x))
      .attr("y1", (d) => yScale(state.data[d.a].y))
      .attr("x2", (d) => xScale(state.data[d.b].x))
      .attr("y2", (d) => yScale(state.data[d.b].y));
  }

  if (model.type === "gmm") {
    layer
      .selectAll("ellipse")
      .data(model.centroids)
      .join("ellipse")
      .attr("class", "density-ring")
      .attr("cx", (d) => xScale(d.x))
      .attr("cy", (d) => yScale(d.y))
      .attr("rx", (_, i) => Math.abs(xScale(model.spreads[i].x) - xScale(0)))
      .attr("ry", (_, i) => Math.abs(yScale(0) - yScale(model.spreads[i].y)))
      .attr("stroke", (_, i) => palette[i % palette.length]);
  }

  if (model.type === "density" || model.type === "optics") {
    layer
      .selectAll("circle.density-ring")
      .data(state.data.filter((_, i) => model.labels[i] !== -1).slice(0, 60))
      .join("circle")
      .attr("class", "density-ring")
      .attr("cx", (d) => xScale(d.x))
      .attr("cy", (d) => yScale(d.y))
      .attr("r", Math.abs(xScale(model.radius || 0.28) - xScale(0)));
  }
}

function drawPoints(svg, model) {
  const points = state.data.map((d, i) => {
    if (state.mode === "unsupervised") {
      const assigned = model.assignments?.[i] ?? model.labels?.[i] ?? model.predict(d);
      return { ...d, assigned, confidence: model.confidences?.[i] ?? 1 };
    }
    const prediction = model.predict(d);
    return { ...d, assigned: d.label, prediction, correct: prediction === d.label };
  });

  const layer = svg.append("g");
  layer
    .selectAll("circle")
    .data(points, (d) => d.id)
    .join("circle")
    .attr("class", (d) => `sample-point${d.isTest ? " test" : ""}${d.isTest && d.correct ? " correct" : ""}${d.isTest && !d.correct ? " wrong" : ""}`)
    .attr("cx", (d) => xScale(d.x))
    .attr("cy", (d) => yScale(d.y))
    .attr("r", (d) => (d.isTest ? 4.3 : 3.6))
    .attr("fill", (d) => (d.assigned === -1 ? noiseColor : palette[Math.abs(d.assigned) % palette.length]))
    .attr("opacity", (d) => (state.mode === "unsupervised" && model.type === "gmm" ? 0.45 + d.confidence * 0.55 : 1));

  if (state.mode === "supervised") {
    layer
      .selectAll("circle.sample-point")
      .filter((d) => d.isTest && !d.correct)
      .attr("r", 5.4);
  }
}

function drawCardChart(card, result) {
  const svg = d3.select(card.querySelector("svg"));
  svg.selectAll("*").remove();
  drawDecisionMap(svg, result.model);
  drawAxes(svg);
  drawOverlays(svg, result.model);
  drawPoints(svg, result.model);
}

function createCard(meta, index) {
  const controls = meta.controls
    .map(
      (control) => `
        <div class="model-control">
          <label>
            ${control.label}
            <output data-param-output="${control.key}"></output>
          </label>
          <input
            type="range"
            min="${control.min}"
            max="${control.max}"
            step="${control.step}"
            data-model="${meta.id}"
            data-param="${control.key}"
          />
        </div>
      `,
    )
    .join("");
  const card = document.createElement("article");
  card.className = "model-card";
  card.dataset.model = meta.id;
  card.style.order = index;
  card.innerHTML = `
    <header class="model-card-header">
      <div>
        <h2>${meta.name}</h2>
        <p class="model-subtitle">${meta.subtitle}</p>
        <a class="model-wiki-link" href="${meta.wiki}" target="_blank" rel="noreferrer">Wikipedia</a>
      </div>
      <div class="score-pill" data-score></div>
    </header>
    <svg class="mini-chart" viewBox="0 0 ${WIDTH} ${HEIGHT}" role="img" aria-label="${meta.name} decision regions"></svg>
    <footer class="model-card-footer">
      <div class="model-controls">${controls}</div>
      <p class="param-line" data-param-line></p>
      <p class="correct-line" data-correct-line></p>
    </footer>
  `;

  card.querySelectorAll("input[type='range']").forEach((input) => {
    input.value = state.modelParams[meta.id][input.dataset.param];
    input.addEventListener("input", (event) => {
      state.modelParams[meta.id][event.target.dataset.param] = Number(event.target.value);
      render();
    });
  });

  return card;
}

function updateCard(card, result, bestScore) {
  const meta = result.meta;
  card.classList.toggle("leader", state.mode === "supervised" && result.metrics.score === bestScore);
  card.querySelector("[data-score]").textContent = result.metrics.label;
  card.querySelector("[data-param-line]").textContent = `${meta.params(result.params)} · ${meta.complexity} complexity`;
  card.querySelector("[data-correct-line]").textContent = result.metrics.detail;

  card.querySelectorAll("input[type='range']").forEach((input) => {
    const control = meta.controls.find((item) => item.key === input.dataset.param);
    const value = result.params[input.dataset.param];
    if (document.activeElement !== input) input.value = value;
    card.querySelector(`[data-param-output="${input.dataset.param}"]`).textContent = Number(value).toFixed(control.digits);
  });

  drawCardChart(card, result);
}

function syncControls() {
  els.sampleValue.textContent = state.samples;
  els.noiseValue.textContent = Number(state.noise).toFixed(2);
  const datasetName = datasetCatalog[state.mode].find((d) => d.id === state.dataset)?.name || "Custom";
  els.readSummary.textContent =
    state.mode === "supervised"
      ? `${datasetName}. Every panel trains on the same generated data. Pale regions show predicted class, filled points show the true class, and outlined dots are holdout examples.`
      : `${datasetName}. Every panel clusters the same generated data. Pale regions show cluster assignment, filled points show assigned clusters, crosses mark centroids when the method has them, and gray marks noise.`;
  els.scoreSummaryLabel.textContent = state.mode === "supervised" ? "Best holdout score" : "Cluster summary";
}

function render() {
  syncControls();
  const results = currentModels().map((meta) => {
    const model = trainModel(meta);
    return { meta, model, params: state.modelParams[meta.id], metrics: evaluate(model) };
  });
  const bestScore = d3.max(results, (d) => d.metrics.score);

  if (state.mode === "supervised") {
    const leaders = results.filter((d) => d.metrics.score === bestScore).map((d) => d.meta.name);
    els.bestModel.textContent = `${leaders.join(", ")} (${bestScore}%)`;
  } else {
    els.bestModel.textContent = `${results.length} clustering models compared`;
  }

  results.forEach((result, index) => {
    let card = cardsByModel.get(result.meta.id);
    if (!card) {
      card = createCard(result.meta, index);
      cardsByModel.set(result.meta.id, card);
      els.modelWall.appendChild(card);
    }
    updateCard(card, result, bestScore);
  });
}

function populateDatasetSelect() {
  els.datasetSelect.innerHTML = currentDatasets().map((item) => `<option value="${item.id}">${item.name}</option>`).join("");
  els.datasetSelect.value = state.dataset;
}

function setMode(mode) {
  state.mode = mode;
  state.dataset = currentDatasets()[0].id;
  document.querySelectorAll(".mode-tab").forEach((button) => button.classList.toggle("active", button.dataset.mode === mode));
  cardsByModel.clear();
  els.modelWall.replaceChildren();
  populateDatasetSelect();
  generateData();
  render();
}

function bindEvents() {
  document.querySelectorAll(".mode-tab").forEach((button) => {
    button.addEventListener("click", () => setMode(button.dataset.mode));
  });

  els.datasetSelect.addEventListener("change", (event) => {
    state.dataset = event.target.value;
    generateData();
    render();
  });

  els.sampleSlider.addEventListener("input", (event) => {
    state.samples = Number(event.target.value);
    generateData();
    render();
  });

  els.noiseSlider.addEventListener("input", (event) => {
    state.noise = Number(event.target.value);
    generateData();
    render();
  });

  els.regenerateButton.addEventListener("click", () => {
    state.seed += 13;
    generateData();
    render();
  });
}

function boot() {
  populateDatasetSelect();
  bindEvents();
  generateData();
  render();
}

boot();
