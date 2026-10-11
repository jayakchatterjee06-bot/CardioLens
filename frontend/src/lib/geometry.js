// Label placement helpers. All functions take plain [x, y, z] arrays so they stay
// independent of React and easy to test.

export function meanPosition(geometry) {
  const pos = geometry.attributes.position
  let x = 0, y = 0, z = 0
  for (let i = 0; i < pos.count; i++) { x += pos.getX(i); y += pos.getY(i); z += pos.getZ(i) }
  return [x / pos.count, y / pos.count, z / pos.count]
}

// Picks the vessel vertex that sticks out furthest horizontally from the heart centre
// while staying near the vessel's mid-height, so each label sits on its own vessel's
// territory instead of crowding at the apex.
export function outwardAnchor(geometry, heartCentre) {
  const [mx, my, mz] = meanPosition(geometry)
  const [cx, , cz] = heartCentre
  const len = Math.hypot(mx - cx, mz - cz) || 1
  const dx = (mx - cx) / len
  const dz = (mz - cz) / len

  const pos = geometry.attributes.position
  let best = 0
  let bestScore = -Infinity
  for (let i = 0; i < pos.count; i++) {
    const score = (pos.getX(i) - cx) * dx + (pos.getZ(i) - cz) * dz - Math.abs(pos.getY(i) - my)
    if (score > bestScore) { bestScore = score; best = i }
  }
  return [pos.getX(best), pos.getY(best), pos.getZ(best)]
}

// True when a surface point faces the camera, judged from the heart centre outward.
export function facesCamera(anchor, heartCentre, cameraPosition) {
  const n = anchor.map((a, i) => a - heartCentre[i])
  return n[0] * (cameraPosition.x - anchor[0]) + n[1] * (cameraPosition.y - anchor[1]) + n[2] * (cameraPosition.z - anchor[2]) > 0
}
