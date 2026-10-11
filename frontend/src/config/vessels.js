// Single mapping between model output keys (ml/config.py TARGETS) and 3D anatomy.
// `key` is also the glTF node name written by tools/build_heart_model.py, so a
// prediction for "lad" can only ever colour the LAD mesh.
export const VESSELS = [
  { key: 'lad', label: 'LAD', name: 'Left Anterior Descending', territory: 'Anterior wall & septum' },
  { key: 'lcx', label: 'LCX', name: 'Left Circumflex', territory: 'Lateral & posterior wall' },
  { key: 'rca', label: 'RCA', name: 'Right Coronary Artery', territory: 'Right ventricle & inferior wall' },
]

export const RISK_BANDS = [
  { max: 0.33, color: '#22c55e', label: 'Low' },
  { max: 0.66, color: '#eab308', label: 'Moderate' },
  { max: 1.01, color: '#ef4444', label: 'High' },
]

export const NO_PREDICTION_COLOR = '#cbd5e1'

export const riskBand = (p) => RISK_BANDS.find((b) => p < b.max)

export const vesselColor = (p) => (p == null ? NO_PREDICTION_COLOR : riskBand(p).color)
