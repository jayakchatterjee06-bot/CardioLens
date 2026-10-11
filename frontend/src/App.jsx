import { useEffect, useState } from 'react'
import Disclaimer from './components/Disclaimer'
import HeartViewer from './components/HeartViewer'
import { getHealth } from './api/client'
import { VESSELS, riskBand } from './config/vessels'

// Temporary until Task #10 wires real /api/predict output into `probabilities`.
const DEMO_PROBABILITIES = { lad: 0.82, lcx: 0.45, rca: 0.18 }

export default function App() {
  const [apiStatus, setApiStatus] = useState('checking…')
  const [selected, setSelected] = useState(null)
  const [probabilities, setProbabilities] = useState({})

  useEffect(() => {
    getHealth().then((h) => setApiStatus(`connected (${h.targets.join(', ')})`))
      .catch(() => setApiStatus('backend not reachable'))
  }, [])

  const demoOn = Object.keys(probabilities).length > 0
  const vessel = VESSELS.find((v) => v.key === selected)
  const p = vessel ? probabilities[vessel.key] : undefined

  return (
    <>
      <Disclaimer />
      <div className="layout">
        <div className="viewer">
          <HeartViewer probabilities={probabilities} selectedVessel={selected} onSelectVessel={setSelected} />
        </div>
        <aside className="panel">
          <h2>CardioLens</h2>
          <p className="muted">API: {apiStatus}</p>

          <button className="btn" onClick={() => setProbabilities(demoOn ? {} : DEMO_PROBABILITIES)}>
            {demoOn ? 'Clear demo colours' : 'Preview colour mapping (demo values)'}
          </button>

          <div className="card">
            {vessel ? (
              <>
                <h3>{vessel.label} — {vessel.name}</h3>
                <p className="muted">Supplies: {vessel.territory}</p>
                <p>
                  Stenosis probability:{' '}
                  {p == null ? 'no prediction yet' : <strong>{Math.round(p * 100)}% ({riskBand(p).label})</strong>}
                </p>
              </>
            ) : (
              <p className="muted">Click a coronary artery (or its label) to inspect vessel-level risk.</p>
            )}
          </div>
        </aside>
      </div>
    </>
  )
}
