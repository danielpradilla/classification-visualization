const { palette } = window.ClassificationVizShared;

document.querySelector("#paletteSwatches").innerHTML = palette
  .map((color, index) => `<div class="swatch" style="background:${color}">${index + 1}<br>${color}</div>`)
  .join("");
