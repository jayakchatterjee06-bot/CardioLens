import { Suspense, useEffect, useMemo, useRef, useState } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, useGLTF } from '@react-three/drei'
import { Vector3 } from 'three'
import { NO_PREDICTION_COLOR, RISK_BANDS, VESSELS, vesselColor } from '../config/vessels'
import { CONTEXT_PARTS, MODEL_CREDIT, MODEL_URL } from '../config/anatomy'
import { facesCamera, meanPosition, outwardAnchor } from '../lib/geometry'

function ContextPart({ geometry, color, roughness, translucent, onClick }) {
  return (
    <mesh geometry={geometry} onClick={onClick}>
      <meshStandardMaterial
        color={color}
        roughness={roughness}
        transparent={translucent}
        opacity={translucent ? 0.18 : 1}
        depthWrite={!translucent}
      />
    </mesh>
  )
}

function VesselMesh({ geometry, color, selected, onSelect }) {
  const [hovered, setHovered] = useState(false)
  return (
    <mesh
      geometry={geometry}
      onClick={(e) => { e.stopPropagation(); onSelect() }}
      onPointerOver={(e) => { e.stopPropagation(); setHovered(true); document.body.style.cursor = 'pointer' }}
      onPointerOut={() => { setHovered(false); document.body.style.cursor = 'auto' }}
    >
      <meshStandardMaterial
        color={color}
        emissive={color}
        emissiveIntensity={selected ? 0.9 : hovered ? 0.5 : 0.2}
        roughness={0.35}
      />
    </mesh>
  )
}

function HeartModel({ probabilities, selectedVessel, onSelectVessel, translucent, onAnchors }) {
  const { nodes } = useGLTF(MODEL_URL)
  const anchors = useMemo(() => {
    const centre = meanPosition(nodes.myocardium.geometry)
    const points = Object.fromEntries(VESSELS.map((v) => [v.key, outwardAnchor(nodes[v.key].geometry, centre)]))
    return { centre, points }
  }, [nodes])
  useEffect(() => onAnchors(anchors), [anchors, onAnchors])

  const deselect = (e) => { e.stopPropagation(); onSelectVessel(null) }

  return (
    <group>
      {CONTEXT_PARTS.map((part) => (
        <ContextPart
          key={part.node}
          geometry={nodes[part.node].geometry}
          color={part.color}
          roughness={part.roughness}
          translucent={translucent && part.node === 'myocardium'}
          onClick={deselect}
        />
      ))}
      {VESSELS.map((v) => (
        <VesselMesh
          key={v.key}
          geometry={nodes[v.key].geometry}
          color={vesselColor(probabilities[v.key])}
          selected={selectedVessel === v.key}
          onSelect={() => onSelectVessel(v.key)}
        />
      ))}
    </group>
  )
}

// Projects 3D anchors to screen space each frame and moves the DOM labels directly (no React re-render).
function LabelProjector({ anchors, labelEls }) {
  const v = useMemo(() => new Vector3(), [])
  useFrame(({ camera, size }) => {
    const { centre, points = {} } = anchors.current
    for (const [key, anchor] of Object.entries(points)) {
      const el = labelEls.current[key]
      if (!el) continue
      v.set(...anchor).project(camera)
      const x = ((v.x + 1) / 2) * size.width
      const y = ((1 - v.y) / 2) * size.height
      el.style.transform = `translate(${x}px, ${y}px) translate(-50%, -50%)`
      el.style.visibility = v.z < 1 ? 'visible' : 'hidden'
      el.classList.toggle('far-side', !facesCamera(anchor, centre, camera.position))
    }
  })
  return null
}

export default function HeartViewer({ probabilities = {}, selectedVessel = null, onSelectVessel = () => {} }) {
  const [translucent, setTranslucent] = useState(false)
  const [modelReady, setModelReady] = useState(false)
  const anchors = useRef({})
  const labelEls = useRef({})

  const handleAnchors = useMemo(() => (a) => { anchors.current = a; setModelReady(true) }, [])

  return (
    <div className="viewer-shell">
      <Canvas camera={{ position: [0, 0.2, 4.2], fov: 40 }} dpr={[1, 2]} onPointerMissed={() => onSelectVessel(null)}>
        <color attach="background" args={['#0f1419']} />
        <hemisphereLight args={['#ffffff', '#3a2a2a', 0.9]} />
        <directionalLight position={[3, 4, 5]} intensity={1.6} />
        <directionalLight position={[-4, -2, -3]} intensity={0.5} />
        <Suspense fallback={null}>
          <HeartModel
            probabilities={probabilities}
            selectedVessel={selectedVessel}
            onSelectVessel={onSelectVessel}
            translucent={translucent}
            onAnchors={handleAnchors}
          />
        </Suspense>
        <LabelProjector anchors={anchors} labelEls={labelEls} />
        <OrbitControls enablePan={false} minDistance={2} maxDistance={8} enableDamping />
      </Canvas>

      {!modelReady && <div className="viewer-loading">Loading the heart model…</div>}

      {modelReady && VESSELS.map((v) => {
        const p = probabilities[v.key]
        return (
          <button
            key={v.key}
            ref={(el) => { labelEls.current[v.key] = el }}
            className={`vessel-label${selectedVessel === v.key ? ' selected' : ''}`}
            style={{ borderColor: vesselColor(p) }}
            onClick={() => onSelectVessel(v.key)}
          >
            {v.label}{p != null && <span> {Math.round(p * 100)}%</span>}
          </button>
        )
      })}

      <div className="viewer-overlay top-right">
        <label className="toggle">
          <input type="checkbox" checked={translucent} onChange={(e) => setTranslucent(e.target.checked)} />
          See-through myocardium
        </label>
      </div>
      <div className="viewer-overlay bottom-left">
        <div className="legend">
          {RISK_BANDS.map((b) => <span key={b.label}><i style={{ background: b.color }} />{b.label}</span>)}
          <span><i style={{ background: NO_PREDICTION_COLOR }} />No prediction</span>
        </div>
        <div className="credit">{MODEL_CREDIT} · Drag to rotate · Scroll to zoom · Click a vessel</div>
      </div>
    </div>
  )
}

useGLTF.preload(MODEL_URL)
