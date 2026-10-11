import { useState } from 'react'
import { GROUP_LABELS, displayValue } from '../lib/format'

function FieldInput({ feature, value, onChange }) {
  const id = `field-${feature.name}`
  if (feature.kind === 'continuous') {
    return (
      <input
        id={id}
        type="number"
        value={value === '' || value == null ? '' : value}
        min={feature.min}
        max={feature.max}
        step="any"
        onChange={(e) => onChange(e.target.value === '' ? '' : Number(e.target.value))}
      />
    )
  }
  // Categorical: select over declared levels; values keep their original type (0/1 vs "N"/"Y").
  const index = feature.categories.findIndex((c) => c === value)
  return (
    <select id={id} value={index} onChange={(e) => onChange(feature.categories[Number(e.target.value)])}>
      {index === -1 && <option value={-1}>Choose</option>}
      {feature.categories.map((level, i) => (
        <option key={String(level)} value={i}>{displayValue(feature, level)}</option>
      ))}
    </select>
  )
}

function Field({ feature, value, error, onChange }) {
  return (
    <div className={`field${error ? ' has-error' : ''}`}>
      <label htmlFor={`field-${feature.name}`}>
        {feature.label}
        {feature.unit && <span className="unit">{feature.unit}</span>}
      </label>
      <FieldInput feature={feature} value={value} onChange={onChange} />
      {error && <p className="field-error">{error}</p>}
      {!error && feature.kind === 'continuous' && (
        <p className="field-hint">{feature.min}–{feature.max}</p>
      )}
    </div>
  )
}

function Group({ name, features, patient, fieldErrors, onChange, defaultOpen }) {
  const [open, setOpen] = useState(defaultOpen)
  const errorCount = features.filter((f) => fieldErrors[f.name]).length
  return (
    <section className="group">
      <button type="button" className="group-toggle" aria-expanded={open} onClick={() => setOpen(!open)}>
        <span className="chevron" aria-hidden="true">{open ? '▾' : '▸'}</span>
        {GROUP_LABELS[name] ?? name}
        <span className="group-meta">{errorCount ? `${errorCount} to fix` : `${features.length}`}</span>
      </button>
      {open && (
        <div className="group-fields">
          {features.map((f) => (
            <Field key={f.name} feature={f} value={patient[f.name]} error={fieldErrors[f.name]} onChange={(v) => onChange(f.name, v)} />
          ))}
        </div>
      )}
    </section>
  )
}

export default function PatientForm({ metadata, samples, patient, sampleId, fieldErrors, onLoadSample, onChange, onPredict, busy }) {
  if (!metadata || !patient) return <p className="muted">Loading the clinical feature list…</p>

  const groups = []
  for (const feature of metadata.features) {
    let group = groups.find((g) => g.name === feature.group)
    if (!group) groups.push((group = { name: feature.group, features: [] }))
    group.features.push(feature)
  }
  const activeSample = samples.find((s) => s.id === sampleId)

  return (
    <form className="patient-form" onSubmit={(e) => { e.preventDefault(); onPredict() }}>
      <h2>Patient</h2>
      <p className="muted">Load a real case, then change any value to see the heart respond.</p>

      <div className="sample-buttons">
        {samples.map((s) => (
          <button
            key={s.id}
            type="button"
            className={`sample${sampleId === s.id ? ' active' : ''}`}
            title={s.description}
            onClick={() => onLoadSample(s)}
          >
            {s.title}
          </button>
        ))}
      </div>
      {activeSample ? (
        <p className="sample-note">{activeSample.description} {activeSample.note}</p>
      ) : (
        <p className="sample-note">Edited values. The prediction updates as you type.</p>
      )}

      {groups.map((g, i) => (
        <Group key={g.name} name={g.name} features={g.features} patient={patient} fieldErrors={fieldErrors} onChange={onChange} defaultOpen={i === 0} />
      ))}

      <button type="submit" className="btn primary" disabled={busy}>{busy ? 'Predicting…' : 'Predict'}</button>
    </form>
  )
}
