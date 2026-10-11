import { VESSELS, riskBand } from '../config/vessels'
import { meanStd, modelName, pct } from '../lib/format'

function RiskChip({ probability }) {
  const band = riskBand(probability)
  return <span className="chip" style={{ '--chip': band.color }}>{band.label} risk</span>
}

function thresholdText(prediction) {
  const t = pct(prediction.threshold)
  return prediction.predicted_positive ? `above the ${t} decision threshold` : `below the ${t} decision threshold`
}

function VesselRow({ vessel, prediction, selected, onSelect }) {
  const p = prediction?.probability
  return (
    <button type="button" className={`vessel-row${selected ? ' selected' : ''}`} onClick={() => onSelect(selected ? null : vessel.key)}>
      <span className="vessel-row-name">
        <strong>{vessel.label}</strong> {vessel.name}
      </span>
      {p == null ? <span className="muted">—</span> : (
        <>
          <span className="vessel-row-pct">{pct(p)}</span>
          <RiskChip probability={p} />
        </>
      )}
    </button>
  )
}

export function PredictionCards({ result, selectedVessel, onSelectVessel }) {
  const predictions = result?.predictions
  const cad = predictions?.cad
  return (
    <section className="results">
      <h2>Prediction</h2>
      {cad ? (
        <div className="cad-result">
          <div className="cad-headline">
            <span className="cad-pct">{pct(cad.probability)}</span>
            <span className="cad-text">
              probability of coronary artery disease
              <br />
              <RiskChip probability={cad.probability} /> <span className="muted">{thresholdText(cad)}</span>
            </span>
          </div>
        </div>
      ) : (
        <p className="muted">Load or enter a patient to get a prediction.</p>
      )}

      <h3>Stenosis by vessel</h3>
      <div className="vessel-rows">
        {VESSELS.map((v) => (
          <VesselRow key={v.key} vessel={v} prediction={predictions?.[v.key]} selected={selectedVessel === v.key} onSelect={onSelectVessel} />
        ))}
      </div>
      <p className="muted small">Select a vessel here or on the heart to see what drives its risk.</p>
    </section>
  )
}

export function ModelPerformance({ metadata }) {
  if (!metadata) return null
  const { validation, dataset } = metadata
  return (
    <section className="performance">
      <h3>How the models performed</h3>
      <p className="muted small">
        Repeated stratified {validation.n_splits}-fold cross-validation ({validation.n_repeats} repeats) on {dataset.rows} patients.
        ROC-AUC is mean ± standard deviation across folds; accuracy and recall are out-of-fold at the {pct(metadata.decision_threshold)} threshold.
      </p>
      <table className="metrics">
        <thead>
          <tr><th>Target</th><th>Model</th><th>ROC-AUC</th><th>Accuracy</th><th>Recall</th></tr>
        </thead>
        <tbody>
          {metadata.targets.map((t) => (
            <tr key={t.key}>
              <td>{t.display_name}</td>
              <td>{modelName(t.model)}</td>
              <td>{meanStd(t.cv_metrics.roc_auc)}</td>
              <td>{t.oof_metrics_calibrated.accuracy.toFixed(2)}</td>
              <td>{t.oof_metrics_calibrated.recall.toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  )
}
