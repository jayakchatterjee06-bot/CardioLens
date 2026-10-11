import { useMemo, useState } from 'react'
import Disclaimer from './components/Disclaimer'
import HeartViewer from './components/HeartViewer'
import PatientForm from './components/PatientForm'
import ShapChart from './components/ShapChart'
import { ModelPerformance, PredictionCards } from './components/Dashboard'
import { usePrediction } from './hooks/usePrediction'
import { VESSELS } from './config/vessels'

export default function App() {
  const { metadata, samples, patient, sampleId, result, busy, error, fieldErrors, loadSample, setField, predictNow } = usePrediction()
  const [selectedVessel, setSelectedVessel] = useState(null)

  const probabilities = useMemo(() => {
    if (!result) return {}
    return Object.fromEntries(VESSELS.map((v) => [v.key, result.predictions[v.key]?.probability]))
  }, [result])

  const featuresByName = useMemo(
    () => Object.fromEntries((metadata?.features ?? []).map((f) => [f.name, f])),
    [metadata],
  )

  const explained = selectedVessel ?? 'cad'
  const explainedTitle = selectedVessel ? VESSELS.find((v) => v.key === selectedVessel).name : 'overall CAD'

  return (
    <>
      <Disclaimer />
      <div className="layout">
        <aside className="column column-form">
          <PatientForm
            metadata={metadata}
            samples={samples}
            patient={patient}
            sampleId={sampleId}
            fieldErrors={fieldErrors}
            onLoadSample={loadSample}
            onChange={setField}
            onPredict={predictNow}
            busy={busy}
          />
        </aside>

        <main className={`viewer${busy ? ' busy' : ''}`}>
          <HeartViewer probabilities={probabilities} selectedVessel={selectedVessel} onSelectVessel={setSelectedVessel} />
          {busy && <div className="viewer-status">Updating…</div>}
        </main>

        <aside className="column column-results">
          {error && <div className="error-banner" role="alert">{error}</div>}
          <PredictionCards result={result} selectedVessel={selectedVessel} onSelectVessel={setSelectedVessel} />
          <ShapChart prediction={result?.predictions[explained]} title={explainedTitle} featuresByName={featuresByName} />
          <ModelPerformance metadata={metadata} />
        </aside>
      </div>
    </>
  )
}
