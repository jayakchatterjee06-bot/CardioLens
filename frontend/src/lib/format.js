export const MODEL_NAMES = {
  logistic_regression: 'Logistic regression',
  random_forest: 'Random forest',
  gradient_boosting: 'Gradient boosting',
}

export const GROUP_LABELS = {
  demographic: 'Demographics',
  history: 'History and risk factors',
  examination: 'Symptoms and examination',
  ecg: 'ECG',
  laboratory: 'Laboratory',
  echocardiography: 'Echocardiography',
}

export const modelName = (key) => MODEL_NAMES[key] ?? key
export const pct = (p, digits = 0) => `${(p * 100).toFixed(digits)}%`
export const points = (c) => `${c >= 0 ? '+' : '−'}${Math.abs(c * 100).toFixed(1)}`
export const meanStd = (m) => `${m.mean.toFixed(2)} ± ${m.std.toFixed(2)}`

// How a stored categorical value reads to a person.
export function displayValue(feature, value) {
  if (feature.kind === 'binary') {
    const [negative, positive] = feature.categories
    const yesNo = (negative === 0 && positive === 1) || (negative === 'N' && positive === 'Y')
    if (yesNo) return value === positive ? 'Yes' : 'No'
  }
  if (value === 'N' && feature.kind !== 'continuous') return 'None'
  return typeof value === 'number' ? formatNumber(value) : String(value)
}

export function formatNumber(value) {
  return Number.isInteger(value) ? String(value) : value.toFixed(1)
}
