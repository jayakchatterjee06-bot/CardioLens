import { useCallback, useEffect, useRef, useState } from 'react'
import { describeError, fieldErrorsFrom, getMetadata, getSamples, predict } from '../api/client'

const DEBOUNCE_MS = 400

// Dataset floats such as BMI 28.621470374 are shown to two decimals; the change is far below clinical precision.
function tidyNumbers(patient) {
  return Object.fromEntries(
    Object.entries(patient).map(([k, v]) => [k, typeof v === 'number' && !Number.isInteger(v) ? Math.round(v * 100) / 100 : v]),
  )
}

function missingValues(patient) {
  const errors = {}
  for (const [name, value] of Object.entries(patient)) {
    if (value === '' || value == null || Number.isNaN(value)) errors[name] = 'Enter a value'
  }
  return errors
}

/**
 * Owns everything that flows between the form, the heart and the dashboard:
 * reference data from the API, the current patient, and the latest prediction.
 */
export function usePrediction() {
  const [metadata, setMetadata] = useState(null)
  const [samples, setSamples] = useState([])
  const [patient, setPatient] = useState(null)
  const [sampleId, setSampleId] = useState(null) // which sample the current values came from, if any
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [fieldErrors, setFieldErrors] = useState({})

  const requestCounter = useRef(0)
  const debounceTimer = useRef(null)

  const runPrediction = useCallback(async (values) => {
    const blanks = missingValues(values)
    if (Object.keys(blanks).length) { setFieldErrors(blanks); return }

    const requestId = ++requestCounter.current
    setBusy(true)
    try {
      const response = await predict(values)
      if (requestId !== requestCounter.current) return // a newer request is in flight
      setResult(response)
      setError(null)
      setFieldErrors({})
    } catch (err) {
      if (requestId !== requestCounter.current) return
      setFieldErrors(fieldErrorsFrom(err))
      setError(describeError(err))
    } finally {
      if (requestId === requestCounter.current) setBusy(false)
    }
  }, [])

  const loadSample = useCallback((sample) => {
    clearTimeout(debounceTimer.current)
    setPatient(tidyNumbers(sample.patient))
    setSampleId(sample.id)
    runPrediction(sample.patient)
  }, [runPrediction])

  const setField = useCallback((name, value) => {
    setSampleId(null)
    setPatient((current) => {
      const next = { ...current, [name]: value }
      clearTimeout(debounceTimer.current)
      debounceTimer.current = setTimeout(() => runPrediction(next), DEBOUNCE_MS)
      return next
    })
  }, [runPrediction])

  const predictNow = useCallback(() => {
    clearTimeout(debounceTimer.current)
    if (patient) runPrediction(patient)
  }, [patient, runPrediction])

  useEffect(() => {
    let cancelled = false
    Promise.all([getMetadata(), getSamples()])
      .then(([meta, sampleList]) => {
        if (cancelled) return
        setMetadata(meta)
        setSamples(sampleList)
        if (sampleList.length) loadSample(sampleList[0])
      })
      .catch((err) => { if (!cancelled) setError(describeError(err)) })
    return () => { cancelled = true; clearTimeout(debounceTimer.current) }
  }, [loadSample])

  return { metadata, samples, patient, sampleId, result, busy, error, fieldErrors, loadSample, setField, predictNow }
}
