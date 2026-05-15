const WIDTH = 820;
const HEIGHT = 560;
const PAD = 42;
const PLOT = {
  left: PAD,
  right: WIDTH - PAD,
  top: PAD,
  bottom: HEIGHT - PAD,
};

const { palette, noiseColor, modelCatalog, datasetCatalog, sortedByName } = window.ClassificationVizShared;
const xScale = d3.scaleLinear().domain([-3, 3]).range([PLOT.left, PLOT.right]);
const yScale = d3.scaleLinear().domain([-3, 3]).range([PLOT.bottom, PLOT.top]);
const state = {
  mode: "supervised",
  model: "logistic",
  dataset: "blobs",
  samples: 180,
  noise: 0.12,
  k: 5,
  depth: 3,
  radius: 0.28,
  kernelWidth: 0.65,
  regularization: 1,
  hiddenUnits: 9,
  nnRate: 0.035,
  nnEpochs: 180,
  forestTrees: 9,
  boostRate: 0.62,
  boostRounds: 6,
  gmmShape: "elliptical",
  gmmAssignment: "soft",
  graphNeighbors: 8,
  minClusterSize: 8,
  seed: 7,
  step: 1,
  data: [],
};

const svg = d3.select("#chart");
const layers = {
  background: svg.append("g"),
  axes: svg.append("g"),
  overlays: svg.append("g"),
  points: svg.append("g"),
};

const els = {
  modelSelect: document.querySelector("#modelSelect"),
  datasetSelect: document.querySelector("#datasetSelect"),
  sampleSlider: document.querySelector("#sampleSlider"),
  noiseSlider: document.querySelector("#noiseSlider"),
  kSlider: document.querySelector("#kSlider"),
  depthSlider: document.querySelector("#depthSlider"),
  radiusSlider: document.querySelector("#radiusSlider"),
  kernelSlider: document.querySelector("#kernelSlider"),
  regularizationSlider: document.querySelector("#regularizationSlider"),
  hiddenUnitsSlider: document.querySelector("#hiddenUnitsSlider"),
  nnRateSlider: document.querySelector("#nnRateSlider"),
  nnEpochsSlider: document.querySelector("#nnEpochsSlider"),
  forestTreesSlider: document.querySelector("#forestTreesSlider"),
  boostRateSlider: document.querySelector("#boostRateSlider"),
  boostRoundsSlider: document.querySelector("#boostRoundsSlider"),
  gmmShapeSelect: document.querySelector("#gmmShapeSelect"),
  gmmAssignmentSelect: document.querySelector("#gmmAssignmentSelect"),
  graphNeighborsSlider: document.querySelector("#graphNeighborsSlider"),
  minClusterSizeSlider: document.querySelector("#minClusterSizeSlider"),
  sampleValue: document.querySelector("#sampleValue"),
  noiseValue: document.querySelector("#noiseValue"),
  kLabelText: document.querySelector("#kLabelText"),
  kValue: document.querySelector("#kValue"),
  depthValue: document.querySelector("#depthValue"),
  radiusValue: document.querySelector("#radiusValue"),
  kernelValue: document.querySelector("#kernelValue"),
  regularizationValue: document.querySelector("#regularizationValue"),
  hiddenUnitsValue: document.querySelector("#hiddenUnitsValue"),
  nnRateValue: document.querySelector("#nnRateValue"),
  nnEpochsValue: document.querySelector("#nnEpochsValue"),
  forestTreesValue: document.querySelector("#forestTreesValue"),
  boostRateValue: document.querySelector("#boostRateValue"),
  boostRoundsValue: document.querySelector("#boostRoundsValue"),
  graphNeighborsValue: document.querySelector("#graphNeighborsValue"),
  minClusterSizeValue: document.querySelector("#minClusterSizeValue"),
  modelTitle: document.querySelector("#modelTitle"),
  modelSubtitle: document.querySelector("#modelSubtitle"),
  modelWikiLink: document.querySelector("#modelWikiLink"),
  readText: document.querySelector("#readText"),
  modelNotes: document.querySelector("#modelNotes"),
  useCaseText: document.querySelector("#useCaseText"),
  metricFit: document.querySelector("#metricFit"),
  metricComplexity: document.querySelector("#metricComplexity"),
  metricStep: document.querySelector("#metricStep"),
  legend: document.querySelector("#legend"),
  kControl: document.querySelector("#kControl"),
  depthControl: document.querySelector("#depthControl"),
  radiusControl: document.querySelector("#radiusControl"),
  kernelControl: document.querySelector("#kernelControl"),
  regularizationControl: document.querySelector("#regularizationControl"),
  hiddenUnitsControl: document.querySelector("#hiddenUnitsControl"),
  nnRateControl: document.querySelector("#nnRateControl"),
  nnEpochsControl: document.querySelector("#nnEpochsControl"),
  forestTreesControl: document.querySelector("#forestTreesControl"),
  boostRateControl: document.querySelector("#boostRateControl"),
  boostRoundsControl: document.querySelector("#boostRoundsControl"),
  gmmShapeControl: document.querySelector("#gmmShapeControl"),
  gmmAssignmentControl: document.querySelector("#gmmAssignmentControl"),
  graphNeighborsControl: document.querySelector("#graphNeighborsControl"),
  minClusterSizeControl: document.querySelector("#minClusterSizeControl"),
};

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
    } else if (state.dataset === "concentric-blobs") {
      const inner = i < n * 0.45;
      const t = random() * Math.PI * 2;
      const r = inner ? Math.abs(gaussian(random)) * (0.24 + noise) : 1.55 + gaussian(random) * (0.18 + noise);
      x = Math.cos(t) * r + gaussian(random) * noise * 0.35;
      y = Math.sin(t) * r + gaussian(random) * noise * 0.35;
      label = inner ? 0 : 1;
    } else if (state.dataset === "bridge") {
      const bucket = random();
      if (bucket < 0.42) {
        x = -1.65 + gaussian(random) * (0.32 + noise);
        y = -0.25 + gaussian(random) * (0.34 + noise);
        label = 0;
      } else if (bucket < 0.84) {
        x = 1.65 + gaussian(random) * (0.32 + noise);
        y = 0.35 + gaussian(random) * (0.34 + noise);
        label = 1;
      } else {
        const t = random();
        x = -1.55 + t * 3.1 + gaussian(random) * (0.08 + noise * 0.7);
        y = -0.2 + t * 0.55 + gaussian(random) * (0.08 + noise * 0.7);
        label = 2;
      }
    } else if (state.dataset === "elongated") {
      const cluster = i % 3;
      const centers = [
        { x: -1.45, y: -0.95, angle: 0.72, long: 0.95, short: 0.16 },
        { x: 1.25, y: -0.45, angle: -0.55, long: 1.05, short: 0.18 },
        { x: -0.05, y: 1.25, angle: 0.08, long: 1.2, short: 0.14 },
      ];
      const c = centers[cluster];
      const u = gaussian(random);
      const v = gaussian(random);
      x = c.x + Math.cos(c.angle) * u * (c.long + noise) - Math.sin(c.angle) * v * (c.short + noise * 0.6);
      y = c.y + Math.sin(c.angle) * u * (c.long + noise) + Math.cos(c.angle) * v * (c.short + noise * 0.6);
      label = cluster;
    } else if (state.dataset === "outliers") {
      const outlier = random() > 0.82;
      if (outlier) {
        x = -2.65 + random() * 5.3;
        y = -2.45 + random() * 4.9;
        label = -1;
      } else {
        const cluster = i % 3;
        const centers = [
          { x: -1.35, y: 0.95, s: 0.3 },
          { x: 1.3, y: 0.85, s: 0.34 },
          { x: 0.05, y: -1.25, s: 0.38 },
        ];
        const c = centers[cluster];
        x = c.x + gaussian(random) * (c.s + noise);
        y = c.y + gaussian(random) * (c.s + noise);
        label = cluster;
      }
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

function trainModel(data) {
  const train = data.filter((d) => !d.isTest);
  if (state.mode === "unsupervised") {
    if (state.model === "dbscan") return trainDbscan(data);
    if (state.model === "spectral") return trainSpectral(data);
    if (state.model === "optics") return trainOptics(data);
    if (state.model === "gmm") return trainGmm(data);
    if (state.model === "hierarchical") return trainHierarchical(data);
    return trainKmeans(data);
  }

  if (state.model === "knn") return trainKnn(train);
  if (state.model === "svm") return trainLinear(train, "svm");
  if (state.model === "rbf-svm") return trainRbfSvm(train);
  if (state.model === "neural-net") return trainNeuralNet(train);
  if (state.model === "lda") return trainDiscriminant(train, "lda");
  if (state.model === "qda") return trainDiscriminant(train, "qda");
  if (state.model === "tree") return trainTree(train, state.depth);
  if (state.model === "forest") return trainForest(train, state.depth);
  if (state.model === "xgboost") return trainBoostedTrees(train);
  if (state.model === "bayes") return trainBayes(train);
  return trainLinear(train, "logistic");
}

function trainLinear(train, kind) {
  let w0 = 0;
  let w1 = 0;
  let b = 0;
  const rate = kind === "svm" ? 0.012 : 0.055;
  const c = state.regularization;
  const shrink = 0.01 / Math.max(0.1, c);
  const loops = 120 + state.step * 40;
  for (let iter = 0; iter < loops; iter += 1) {
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
  return {
    type: "knn",
    predict: (p) => {
      const k = Math.min(state.k, train.length);
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
  const gamma = 1 / (2 * state.kernelWidth ** 2);
  const epochs = 3 + state.step;
  const c = state.regularization;
  const alpha = new Array(train.length).fill(0);
  const targets = train.map((d) => (d.label === 1 ? 1 : -1));
  const kernel = (a, b) => Math.exp(-gamma * dist2(a, b));

  for (let epoch = 0; epoch < epochs; epoch += 1) {
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
  const hiddenCount = state.hiddenUnits;
  const hidden = d3.range(hiddenCount).map(() => ({
    wx: gaussian(random) * 0.55,
    wy: gaussian(random) * 0.55,
    b: gaussian(random) * 0.12,
  }));
  const output = d3.range(hiddenCount).map(() => gaussian(random) * 0.45);
  let outputBias = 0;
  const rate = state.nnRate;
  const loops = state.nnEpochs;

  for (let iter = 0; iter < loops; iter += 1) {
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

function covarianceStats(rows) {
  const meanX = d3.mean(rows, (d) => d.x) || 0;
  const meanY = d3.mean(rows, (d) => d.y) || 0;
  const denom = Math.max(1, rows.length - 1);
  const xx = d3.sum(rows, (d) => (d.x - meanX) ** 2) / denom + 0.08;
  const yy = d3.sum(rows, (d) => (d.y - meanY) ** 2) / denom + 0.08;
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
  const groups = d3.group(train, (d) => d.label);
  const classStats = [...groups.entries()].map(([label, rows]) => ({
    label,
    prior: rows.length / train.length,
    covariance: covarianceStats(rows),
    rows,
  }));

  if (kind === "lda") {
    const pooled = covarianceStats(train);
    classStats.forEach((stat) => {
      stat.covariance = {
        ...pooled,
        meanX: covarianceStats(stat.rows).meanX,
        meanY: covarianceStats(stat.rows).meanY,
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
  const treeCount = state.forestTrees;
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
  const learningRate = state.boostRate;
  const positiveRate = clamp(d3.mean(train, (d) => d.label) || 0.5, 0.05, 0.95);
  const initialScore = Math.log(positiveRate / (1 - positiveRate));
  const scores = new Map(train.map((d) => [d.id, initialScore]));
  const trees = [];
  const rounds = state.boostRounds;
  const depth = Math.min(2, Math.max(1, state.depth));

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
  const groups = d3.group(train, (d) => d.label);
  const stats = [...groups.entries()].map(([label, rows]) => {
    const meanX = d3.mean(rows, (d) => d.x);
    const meanY = d3.mean(rows, (d) => d.y);
    const varX = d3.variance(rows, (d) => d.x) || 0.1;
    const varY = d3.variance(rows, (d) => d.y) || 0.1;
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

function trainKmeans(data) {
  const k = clamp(state.k, 2, 12);
  let centroids = d3.range(k).map((i) => {
    const angle = (i / k) * Math.PI * 2;
    const ring = i < 8 ? 1.35 : 2.15;
    return { x: Math.cos(angle) * ring, y: Math.sin(angle) * ring, label: i };
  });
  const iterations = 2 + state.step * 2;
  let assignments = [];

  for (let iter = 0; iter < iterations; iter += 1) {
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

function nearestCentroid(p, centroids) {
  return centroids
    .map((c, i) => ({ i, distance: dist2(p, c) }))
    .sort((a, b) => a.distance - b.distance)[0].i;
}

function trainDbscan(data) {
  const radius = state.radius;
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
  const targetClusters = clamp(state.k, 2, 12);
  const graphK = state.graphNeighbors;
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
  const minPts = clamp(state.k, 3, 10);
  const minClusterSize = state.minClusterSize;
  const sortedDistances = data.map((p, i) =>
    data
      .map((d, j) => (i === j ? Infinity : Math.sqrt(dist2(p, d))))
      .sort((a, b) => a - b),
  );
  const coreDistance = sortedDistances.map((distances) => distances[minPts - 1] || Infinity);
  const finiteCoreDistances = coreDistance.filter(Number.isFinite).sort((a, b) => a - b);
  const adaptiveCut = (d3.quantile(finiteCoreDistances, 0.68) || state.radius) * 1.35;
  const reachabilityCut = Math.max(state.radius * 2.3, adaptiveCut);
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
    coreDistance,
    predict: (p) => nearestPointLabel(p, data, labels, reachabilityCut * 1.25),
  };
}

function trainGmm(data) {
  const result = trainKmeans(data);
  const spreads = result.centroids.map((c, i) => {
    const members = data.filter((_, idx) => result.assignments[idx] === i);
    const sx = Math.max(0.18, Math.sqrt(d3.mean(members, (d) => (d.x - c.x) ** 2) || 0.18));
    const sy = Math.max(0.18, Math.sqrt(d3.mean(members, (d) => (d.y - c.y) ** 2) || 0.18));
    const spherical = Math.max(sx, sy);
    return state.gmmShape === "spherical" ? { x: spherical, y: spherical } : { x: sx, y: sy };
  });
  const softResults = data.map((p) => softClusterResult(p, result.centroids, spreads));
  return {
    type: "gmm",
    centroids: result.centroids,
    spreads,
    confidences: softResults.map((d) => d.confidence),
    assignments: state.gmmAssignment === "hard" ? result.assignments : softResults.map((d) => d.label),
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
  const k = clamp(state.k, 2, 12);
  let groups = data.map((p, i) => ({ id: i, points: [p], x: p.x, y: p.y }));
  const target = Math.max(k, 14 - state.step * 2);
  while (groups.length > target) {
    let best = { a: 0, b: 1, distance: Infinity };
    for (let i = 0; i < groups.length; i += 1) {
      for (let j = i + 1; j < groups.length; j += 1) {
        const distance = dist2(groups[i], groups[j]);
        if (distance < best.distance) best = { a: i, b: j, distance };
      }
    }
    const a = groups[best.a];
    const b = groups[best.b];
    const merged = {
      id: a.id,
      points: [...a.points, ...b.points],
      x: d3.mean([...a.points, ...b.points], (d) => d.x),
      y: d3.mean([...a.points, ...b.points], (d) => d.y),
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

function drawAxes() {
  layers.axes.selectAll("*").remove();
  layers.axes
    .append("rect")
    .attr("x", PLOT.left)
    .attr("y", PLOT.top)
    .attr("width", PLOT.right - PLOT.left)
    .attr("height", PLOT.bottom - PLOT.top)
    .attr("fill", "none")
    .attr("stroke", "#cbd5e1");

  layers.axes
    .append("line")
    .attr("x1", xScale(-3))
    .attr("x2", xScale(3))
    .attr("y1", yScale(0))
    .attr("y2", yScale(0))
    .attr("stroke", "#d8dee9");

  layers.axes
    .append("line")
    .attr("x1", xScale(0))
    .attr("x2", xScale(0))
    .attr("y1", yScale(-3))
    .attr("y2", yScale(3))
    .attr("stroke", "#d8dee9");

  layers.axes
    .append("text")
    .attr("class", "axis-label")
    .attr("x", PLOT.right - 40)
    .attr("y", PLOT.bottom + 28)
    .text("feature x");

  layers.axes
    .append("text")
    .attr("class", "axis-label")
    .attr("x", PLOT.left - 30)
    .attr("y", PLOT.top + 8)
    .text("feature y");
}

function drawDecisionMap(model) {
  const cells = [];
  const grid = 42;
  for (let ix = 0; ix < grid; ix += 1) {
    for (let iy = 0; iy < grid; iy += 1) {
      const x = -3 + (ix / grid) * 6;
      const y = -3 + (iy / grid) * 6;
      cells.push({ x, y, label: model.predict({ x, y }) });
    }
  }
  const cellW = (PLOT.right - PLOT.left) / grid + 1;
  const cellH = (PLOT.bottom - PLOT.top) / grid + 1;
  layers.background
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

function drawOverlays(model) {
  layers.overlays.selectAll("*").remove();
  if (model.type === "linear") {
    const points = [-3, 3].map((x) => ({ x, y: -(model.w0 * x + model.b) / (model.w1 || 0.0001) }));
    layers.overlays
      .append("path")
      .attr("class", "boundary-line")
      .attr("d", d3.line().x((d) => xScale(d.x)).y((d) => yScale(d.y))(points));
  }

  if (model.type === "kernel") {
    layers.overlays
      .selectAll("circle")
      .data(model.supportVectors.slice(0, 90))
      .join("circle")
      .attr("class", "support-vector")
      .attr("cx", (d) => xScale(d.x))
      .attr("cy", (d) => yScale(d.y))
      .attr("r", 8.5)
      .attr("fill", "none");
  }

  if (model.type === "tree" || model.type === "boosted") {
    layers.overlays
      .selectAll("line")
      .data(model.type === "boosted" ? model.splits.slice(0, 16) : model.splits)
      .join("line")
      .attr("class", "tree-split")
      .attr("x1", (d) => (d.feature === "x" ? xScale(d.threshold) : xScale(d.bounds.xMin)))
      .attr("x2", (d) => (d.feature === "x" ? xScale(d.threshold) : xScale(d.bounds.xMax)))
      .attr("y1", (d) => (d.feature === "x" ? yScale(d.bounds.yMin) : yScale(d.threshold)))
      .attr("y2", (d) => (d.feature === "x" ? yScale(d.bounds.yMax) : yScale(d.threshold)))
      .attr("opacity", (d) => (model.type === "boosted" ? Math.max(0.16, 0.7 - d.learner * 0.05) : 0.7));
  }

  if (["centroid", "gmm", "hierarchical", "graph", "optics"].includes(model.type)) {
    layers.overlays
      .selectAll("path.centroid")
      .data(model.centroids)
      .join("path")
      .attr("class", "centroid")
      .attr("d", d3.symbol().type(d3.symbolCross).size(150))
      .attr("transform", (d) => `translate(${xScale(d.x)},${yScale(d.y)})`)
      .attr("fill", (d, i) => palette[i % palette.length]);
  }

  if (model.type === "graph" || model.type === "optics") {
    layers.overlays
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
    layers.overlays
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
    layers.overlays
      .selectAll("circle")
      .data(state.data.filter((_, i) => model.labels[i] !== -1).slice(0, 80))
      .join("circle")
      .attr("class", "density-ring")
      .attr("cx", (d) => xScale(d.x))
      .attr("cy", (d) => yScale(d.y))
      .attr("r", Math.abs(xScale(state.radius) - xScale(0)));
  }
}

function drawPoints(model) {
  const points = state.data.map((d, i) => {
    const assigned = state.mode === "supervised" ? d.label : model.assignments?.[i] ?? model.labels?.[i] ?? model.predict(d);
    const confidence = model.confidences?.[i] ?? 1;
    return { ...d, assigned, confidence };
  });

  layers.points
    .selectAll("circle")
    .data(points, (d) => d.id)
    .join("circle")
    .attr("class", (d) => `sample-point${d.isTest ? " test" : ""}`)
    .attr("cx", (d) => xScale(d.x))
    .attr("cy", (d) => yScale(d.y))
    .attr("r", (d) => (d.isTest ? 5.8 : 5.2))
    .attr("fill", (d) => (d.assigned === -1 ? noiseColor : palette[Math.abs(d.assigned) % palette.length]))
    .attr("opacity", (d) => (model.type === "gmm" && state.gmmAssignment === "soft" ? 0.45 + d.confidence * 0.55 : 1));

  layers.points
    .selectAll("circle")
    .selectAll("title")
    .data((d) => [d])
    .join("title")
    .text((d) => `${state.mode === "supervised" ? "Class" : "Cluster"} ${d.assigned}`);
}

function supervisedAccuracy(model) {
  const test = state.data.filter((d) => d.isTest);
  const evalSet = test.length ? test : state.data;
  const correct = evalSet.filter((d) => model.predict(d) === d.label).length;
  return Math.round((correct / evalSet.length) * 100);
}

function clusterScore(model) {
  const assignments = state.data.map((d, i) => model.assignments?.[i] ?? model.labels?.[i] ?? model.predict(d)).filter((v) => v !== -1);
  const unique = new Set(assignments);
  const noise = state.data.length - assignments.length;
  return { clusters: unique.size, noise };
}

function renderExplanation(model) {
  const meta = modelCatalog[state.mode].find((m) => m.id === state.model);
  els.modelTitle.textContent = meta.name;
  els.modelSubtitle.textContent = meta.subtitle;
  els.modelWikiLink.href = meta.wiki;
  els.modelWikiLink.textContent = `Wikipedia: ${meta.name}`;
  els.modelNotes.innerHTML = meta.notes.map((note) => `<li>${note}</li>`).join("");
  els.useCaseText.textContent = meta.useCase;
  els.metricComplexity.textContent = meta.complexity;
  els.metricStep.textContent = String(state.step);

  if (state.mode === "supervised") {
    els.readText.textContent = "The pale background shows what the trained model would predict anywhere in the feature space. Dots are labeled examples; black outlines mark test points used for the visible score.";
    els.metricFit.textContent = `${supervisedAccuracy(model)}% holdout accuracy`;
    renderLegend(["Class A", "Class B", "Test points"]);
  } else {
    const { clusters, noise } = clusterScore(model);
    els.readText.textContent = "The background shows the cluster assignment the algorithm would make for a new point. Crosses are centroids when the method uses them; gray points are treated as noise.";
    els.metricFit.textContent = `${clusters} clusters${noise ? `, ${noise} noise` : ""}`;
    renderLegend(["Cluster", "Centroid", "Noise"]);
  }
}

function renderLegend(items) {
  els.legend.innerHTML = items
    .map((item, i) => {
      const color = item.includes("Noise") ? noiseColor : item.includes("Test") ? "#111827" : palette[i % palette.length];
      return `<span class="legend-item"><span class="legend-swatch" style="background:${color}"></span>${item}</span>`;
    })
    .join("");
}

function kControlLabel() {
  if (state.model === "knn") return "Neighbors";
  if (state.model === "optics") return "Minimum points";
  if (["kmeans", "gmm", "hierarchical", "spectral"].includes(state.model)) return "Clusters";
  return "Neighbors / clusters";
}

function syncControls() {
  els.sampleValue.textContent = state.samples;
  els.noiseValue.textContent = Number(state.noise).toFixed(2);
  els.kLabelText.textContent = kControlLabel();
  els.kValue.textContent = state.k;
  els.depthValue.textContent = state.depth;
  els.radiusValue.textContent = Number(state.radius).toFixed(2);
  els.kernelValue.textContent = Number(state.kernelWidth).toFixed(2);
  els.regularizationValue.textContent = Number(state.regularization).toFixed(2);
  els.hiddenUnitsValue.textContent = state.hiddenUnits;
  els.nnRateValue.textContent = Number(state.nnRate).toFixed(3);
  els.nnEpochsValue.textContent = state.nnEpochs;
  els.forestTreesValue.textContent = state.forestTrees;
  els.boostRateValue.textContent = Number(state.boostRate).toFixed(2);
  els.boostRoundsValue.textContent = state.boostRounds;
  els.graphNeighborsValue.textContent = state.graphNeighbors;
  els.minClusterSizeValue.textContent = state.minClusterSize;

  const modelNeedsK = ["knn", "kmeans", "gmm", "hierarchical", "spectral", "optics"].includes(state.model);
  els.kControl.classList.toggle("hidden", !modelNeedsK);
  els.depthControl.classList.toggle("hidden", !["tree", "forest", "xgboost"].includes(state.model));
  els.radiusControl.classList.toggle("hidden", !["dbscan", "optics"].includes(state.model));
  els.kernelControl.classList.toggle("hidden", state.model !== "rbf-svm");
  els.regularizationControl.classList.toggle("hidden", !["logistic", "svm", "rbf-svm"].includes(state.model));
  els.hiddenUnitsControl.classList.toggle("hidden", state.model !== "neural-net");
  els.nnRateControl.classList.toggle("hidden", state.model !== "neural-net");
  els.nnEpochsControl.classList.toggle("hidden", state.model !== "neural-net");
  els.forestTreesControl.classList.toggle("hidden", state.model !== "forest");
  els.boostRateControl.classList.toggle("hidden", state.model !== "xgboost");
  els.boostRoundsControl.classList.toggle("hidden", state.model !== "xgboost");
  els.gmmShapeControl.classList.toggle("hidden", state.model !== "gmm");
  els.gmmAssignmentControl.classList.toggle("hidden", state.model !== "gmm");
  els.graphNeighborsControl.classList.toggle("hidden", state.model !== "spectral");
  els.minClusterSizeControl.classList.toggle("hidden", state.model !== "optics");
}

function render() {
  syncControls();
  const model = trainModel(state.data);
  drawDecisionMap(model);
  drawOverlays(model);
  drawPoints(model);
  renderExplanation(model);
}

function populateSelect(select, items, selected) {
  select.innerHTML = items.map((item) => `<option value="${item.id}">${item.name}</option>`).join("");
  select.value = selected;
}

function setMode(mode) {
  state.mode = mode;
  state.model = sortedByName(modelCatalog[mode])[0].id;
  state.dataset = sortedByName(datasetCatalog[mode])[0].id;
  state.step = 1;
  document.querySelectorAll(".mode-tab").forEach((button) => button.classList.toggle("active", button.dataset.mode === mode));
  populateSelect(els.modelSelect, sortedByName(modelCatalog[mode]), state.model);
  populateSelect(els.datasetSelect, sortedByName(datasetCatalog[mode]), state.dataset);
  generateData();
  render();
}

function bindEvents() {
  document.querySelectorAll(".mode-tab").forEach((button) => {
    button.addEventListener("click", () => setMode(button.dataset.mode));
  });

  els.modelSelect.addEventListener("change", (event) => {
    state.model = event.target.value;
    state.step = 1;
    render();
  });

  els.datasetSelect.addEventListener("change", (event) => {
    state.dataset = event.target.value;
    state.step = 1;
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

  els.kSlider.addEventListener("input", (event) => {
    state.k = Number(event.target.value);
    render();
  });

  els.depthSlider.addEventListener("input", (event) => {
    state.depth = Number(event.target.value);
    render();
  });

  els.radiusSlider.addEventListener("input", (event) => {
    state.radius = Number(event.target.value);
    render();
  });

  els.kernelSlider.addEventListener("input", (event) => {
    state.kernelWidth = Number(event.target.value);
    render();
  });

  els.regularizationSlider.addEventListener("input", (event) => {
    state.regularization = Number(event.target.value);
    render();
  });

  els.hiddenUnitsSlider.addEventListener("input", (event) => {
    state.hiddenUnits = Number(event.target.value);
    render();
  });

  els.nnRateSlider.addEventListener("input", (event) => {
    state.nnRate = Number(event.target.value);
    render();
  });

  els.nnEpochsSlider.addEventListener("input", (event) => {
    state.nnEpochs = Number(event.target.value);
    render();
  });

  els.forestTreesSlider.addEventListener("input", (event) => {
    state.forestTrees = Number(event.target.value);
    render();
  });

  els.boostRateSlider.addEventListener("input", (event) => {
    state.boostRate = Number(event.target.value);
    render();
  });

  els.boostRoundsSlider.addEventListener("input", (event) => {
    state.boostRounds = Number(event.target.value);
    render();
  });

  els.gmmShapeSelect.addEventListener("change", (event) => {
    state.gmmShape = event.target.value;
    render();
  });

  els.gmmAssignmentSelect.addEventListener("change", (event) => {
    state.gmmAssignment = event.target.value;
    render();
  });

  els.graphNeighborsSlider.addEventListener("input", (event) => {
    state.graphNeighbors = Number(event.target.value);
    render();
  });

  els.minClusterSizeSlider.addEventListener("input", (event) => {
    state.minClusterSize = Number(event.target.value);
    render();
  });

  document.querySelector("#regenerateButton").addEventListener("click", () => {
    state.seed += 13;
    state.step = 1;
    generateData();
    render();
  });

  document.querySelector("#stepButton").addEventListener("click", () => {
    state.step = state.step >= 6 ? 1 : state.step + 1;
    render();
  });
}

function boot() {
  drawAxes();
  bindEvents();
  populateSelect(els.modelSelect, sortedByName(modelCatalog.supervised), state.model);
  populateSelect(els.datasetSelect, sortedByName(datasetCatalog.supervised), state.dataset);
  generateData();
  render();
}

boot();
