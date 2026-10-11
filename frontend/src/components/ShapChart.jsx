import { displayValue, modelName, pct, points } from '../lib/format'

const TOP_N = 10
const RAISES = '#eb6834'
const LOWERS = '#3987e5'

export default function ShapChart({ prediction, title, featuresByName }) {
  if (!prediction) return null
  const top = prediction.contributions.slice(0, TOP_N)
  const rest = prediction.contributions.slice(TOP_N)
  const restTotal = rest.reduce((sum, c) => sum + c.contribution, 0)
  const scale = Math.max(...top.map((c) => Math.abs(c.contribution)), Math.abs(restTotal), 1e-6)

  const rows = [
    ...top.map((c) => ({
      key: c.feature,
      label: c.label,
      value: displayValue(featuresByName[c.feature] ?? { kind: 'continuous', categories: [] }, c.value),
      unit: c.unit,
      contribution: c.contribution,
    })),
    { key: '__rest', label: `${rest.length} other features`, value: '', unit: '', contribution: restTotal, rest: true },
  ]

  return (
    <section className="shap">
      <h3>Why this risk for {title}</h3>
      <p className="shap-framing">
        An average patient scores <strong>{pct(prediction.base_value)}</strong>. This patient scores{' '}
        <strong>{pct(prediction.probability)}</strong>. The bars show how much each measurement moved the estimate,
        in percentage points.
      </p>
      <div className="shap-legend">
        <span><i style={{ background: RAISES }} />raises risk</span>
        <span><i style={{ background: LOWERS }} />lowers risk</span>
      </div>
      <ol className="shap-rows">
        {rows.map((r) => {
          const up = r.contribution >= 0
          const width = `${(Math.abs(r.contribution) / scale) * 50}%`
          return (
            <li key={r.key} className={`shap-row${r.rest ? ' rest' : ''}`} title={`${r.label}: ${points(r.contribution)} points`}>
              <span className="shap-label">
                {r.label}
                {r.value !== '' && <span className="shap-value">{r.value}{r.unit && ` ${r.unit}`}</span>}
              </span>
              <span className="shap-bar">
                <i className="axis" />
                <i className="fill" style={{ width, background: up ? RAISES : LOWERS, [up ? 'left' : 'right']: '50%' }} />
              </span>
              <span className="shap-points">{points(r.contribution)}</span>
            </li>
          )
        })}
      </ol>
      <p className="muted small">{modelName(prediction.model)} model. SHAP values on the calibrated probability scale.</p>
    </section>
  )
}
