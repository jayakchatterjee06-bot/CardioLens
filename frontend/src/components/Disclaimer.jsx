export default function Disclaimer() {
  return (
    <div role="alert" style={{
      height: 56, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10,
      background: '#7f1d1d', color: '#fff', fontWeight: 600, padding: '0 16px', textAlign: 'center',
    }}>
      ⚠ For clinical decision support and educational purposes only. Not a substitute for formal
      diagnostic imaging (e.g. coronary angiography) or professional medical judgement.
    </div>
  )
}
