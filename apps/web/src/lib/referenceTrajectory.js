/**
 * Artemis II reference trajectory generator.
 *
 * Uses same trilaterateSpacecraft() as live positioning.
 * Lateral offsets calibrated to match actual free-return trajectory
 * (~100,000+ km off the Earth-Moon line at midpoint).
 */

import { moonPositionECI, trilaterateSpacecraft, eciToCartesian3 } from "./orbit.js";

const MISSION_DAYS = 10;
const STEPS = 600;
const R_EARTH_KM = 6571;
const MOON_ORBIT_KM = 384400;
const MOON_RADIUS_KM = 1737;
const FLYBY_ALT_KM = 2000;

function smoothStep(t) {
  return t * t * (3 - 2 * t);
}

function earthDistProfile(t) {
  if (t < 0.42) {
    const p = t / 0.42;
    return R_EARTH_KM + (MOON_ORBIT_KM - R_EARTH_KM) * smoothStep(p);
  }
  if (t < 0.50) {
    const p = (t - 0.42) / 0.08;
    return MOON_ORBIT_KM - 20000 * Math.sin(p * Math.PI);
  }
  const p = (t - 0.50) / 0.50;
  return MOON_ORBIT_KM - (MOON_ORBIT_KM - R_EARTH_KM) * smoothStep(p);
}

/**
 * Moon distance profile calibrated for realistic free-return arc.
 *
 * A free-return trajectory curves ~100,000-120,000 km off the
 * Earth-Moon line at midpoint. We model this by making moonDist
 * significantly different from |moonOrbit - earthDist|.
 */
function moonDistProfile(earthDist, t) {
  const onLine = Math.abs(MOON_ORBIT_KM - earthDist);

  if (t < 0.42) {
    // Outbound: large lateral arc (~120,000 km at peak)
    const p = t / 0.42;
    const lateralOffset = 120000 * Math.sin(p * Math.PI);
    return Math.sqrt(onLine * onLine + lateralOffset * lateralOffset);
  }
  if (t < 0.50) {
    // Flyby: close approach to Moon
    const p = (t - 0.42) / 0.08;
    return MOON_RADIUS_KM + FLYBY_ALT_KM + 5000 * (1 - Math.sin(p * Math.PI));
  }
  // Return: slightly wider arc on the other side
  const p = (t - 0.50) / 0.50;
  const lateralOffset = 130000 * Math.sin(p * Math.PI);
  return Math.sqrt(onLine * onLine + lateralOffset * lateralOffset);
}

export function generateReferenceTrajectory(launchTimestamp, Cartesian3) {
  const positions = [];
  const dt = (MISSION_DAYS * 86400) / STEPS;

  for (let i = 0; i <= STEPS; i++) {
    const ts = launchTimestamp + i * dt;
    const t = i / STEPS;

    const earthDist = earthDistProfile(t);
    const moonDist = moonDistProfile(earthDist, t);
    const moonECI = moonPositionECI(ts);
    const craftECI = trilaterateSpacecraft(moonECI, earthDist, moonDist);

    if (craftECI) {
      positions.push(eciToCartesian3(craftECI, Cartesian3));
    }
  }

  return positions;
}

export function generateMoonOrbit(timestampSec, Cartesian3, steps = 200) {
  const positions = [];
  const period = 27.321661 * 86400;
  for (let i = 0; i <= steps; i++) {
    const ts = timestampSec - period / 2 + (period * i) / steps;
    const moonECI = moonPositionECI(ts);
    positions.push(eciToCartesian3(moonECI, Cartesian3));
  }
  return positions;
}

export function estimateLaunchTime(telemetryPoint) {
  if (!telemetryPoint) return null;
  const met = telemetryPoint.met;
  if (!met) return null;

  let totalSeconds = 0;
  const matchNew = met.match(/T\+(\d+)d\s+(\d+):(\d+):(\d+)/);
  if (matchNew) {
    totalSeconds = parseInt(matchNew[1]) * 86400 + parseInt(matchNew[2]) * 3600 +
      parseInt(matchNew[3]) * 60 + parseInt(matchNew[4]);
  } else {
    const matchOld = met.match(/(\d+):(\d+):(\d+):(\d+)/);
    if (matchOld) {
      totalSeconds = parseInt(matchOld[1]) * 86400 + parseInt(matchOld[2]) * 3600 +
        parseInt(matchOld[3]) * 60 + parseInt(matchOld[4]);
    }
  }
  if (totalSeconds === 0) return null;
  return telemetryPoint.timestamp - totalSeconds;
}
