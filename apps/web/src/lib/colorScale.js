/**
 * Velocity → Color mapping for trajectory visualization.
 *
 * Blue (slow, ~0.5 km/s coast) → Cyan → Green → Yellow → Red (fast, ~10 km/s TLI)
 */

import { Color } from "cesium";

// Color stops: [velocity km/s, r, g, b]
const STOPS = [
  [0.5, 0.2, 0.3, 1.0],   // blue — slow coast
  [1.5, 0.0, 0.8, 1.0],   // cyan — outbound coast
  [2.5, 0.0, 1.0, 0.5],   // green — moderate
  [4.0, 1.0, 0.8, 0.0],   // yellow — fast
  [7.0, 1.0, 0.3, 0.0],   // orange — very fast
  [10.0, 1.0, 0.0, 0.0],  // red — TLI burn peak
];

/**
 * Map velocity to a Cesium Color.
 */
export function velocityToColor(vel, alpha = 0.85) {
  if (vel <= STOPS[0][0]) {
    return new Color(STOPS[0][1], STOPS[0][2], STOPS[0][3], alpha);
  }
  if (vel >= STOPS[STOPS.length - 1][0]) {
    const s = STOPS[STOPS.length - 1];
    return new Color(s[1], s[2], s[3], alpha);
  }

  for (let i = 1; i < STOPS.length; i++) {
    if (vel <= STOPS[i][0]) {
      const [v0, r0, g0, b0] = STOPS[i - 1];
      const [v1, r1, g1, b1] = STOPS[i];
      const t = (vel - v0) / (v1 - v0);
      return new Color(
        r0 + (r1 - r0) * t,
        g0 + (g1 - g0) * t,
        b0 + (b1 - b0) * t,
        alpha,
      );
    }
  }

  return new Color(1, 0, 0, alpha);
}

/**
 * Build color-coded polyline segments from trajectory data.
 * Returns array of { positions: [Cartesian3, Cartesian3], color: Color }.
 */
export function buildColorSegments(orionData, eciToCartesian3, Cartesian3) {
  const segments = [];

  for (let i = 1; i < orionData.length; i++) {
    const prev = orionData[i - 1];
    const curr = orionData[i];

    if (!prev.pos_km || !curr.pos_km) continue;

    const vel = curr.vel_kms
      ? Math.sqrt(curr.vel_kms[0] ** 2 + curr.vel_kms[1] ** 2 + curr.vel_kms[2] ** 2)
      : 1.0;

    segments.push({
      positions: [
        eciToCartesian3(prev.pos_km, Cartesian3),
        eciToCartesian3(curr.pos_km, Cartesian3),
      ],
      color: velocityToColor(vel),
    });
  }

  return segments;
}
