const BASE = '/api'

export class ApiError extends Error {
  constructor(status, detail) {
    super(`API ${status}`)
    this.status = status
    this.detail = detail // FastAPI: string, or [{loc, msg, type}] for validation errors
  }
}

async function request(path, options = {}) {
  let res
  try {
    res = await fetch(`${BASE}${path}`, { headers: { 'Content-Type': 'application/json' }, ...options })
  } catch {
    throw new ApiError(0, 'unreachable')
  }
  if (!res.ok) {
    let detail = res.statusText
    try { detail = (await res.json()).detail } catch { /* non-JSON error body */ }
    throw new ApiError(res.status, detail)
  }
  return res.json()
}

export const getHealth = () => request('/health')
export const getMetadata = () => request('/metadata')
export const getSamples = () => request('/samples')
export const predict = (patient) => request('/predict', { method: 'POST', body: JSON.stringify(patient) })

// Maps a 422 body to {fieldName: message}. Non-field errors land under the '' key.
export function fieldErrorsFrom(error) {
  if (!(error instanceof ApiError) || error.status !== 422 || !Array.isArray(error.detail)) return {}
  const errors = {}
  for (const item of error.detail) {
    const field = item.loc?.[1] ?? ''
    errors[field] = item.msg
  }
  return errors
}

export function describeError(error) {
  if (!(error instanceof ApiError)) return 'Something went wrong while predicting.'
  if (error.status === 0) return 'Can\'t reach the prediction server. Start it with `uvicorn backend.main:app --port 8000` and reload.'
  if (error.status === 503) return 'The models aren\'t trained yet. Run `python -m ml.train`, then restart the server.'
  if (error.status === 422) return 'Some values need fixing before a prediction can run.'
  return `The server answered ${error.status}.`
}
