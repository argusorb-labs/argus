/**
 * Artemis II reference trajectory generator.
 *
 * Uses the SAME trilateration method as live positioning (orbit.js)
 * to ensure the reference path passes through where Orion actually is.
 *
 * The distance profiles model a free-return trajectory:
 *   Day 0-4:   Outbound (Earth → Moon, decelerating)
 *   Day 4-5:   Lunar flyby (closest approach ~2000 km)
 *   Day 5-10:  Return (Moon → Earth, accelerating)
 */

import { moonPositionECI, trilaterateSpacecraft, eciToCartesian3 } from "./orbit.js";

const MISSION_DAYS = 10;
const STEPS = 600;
const R_EARTH_KM = 6571; // LEO altitude
const MOON_ORBIT_KM = 384400;
const FLYBY_ALT_KM = 2000; // closest approach
const MOON_RADIUS_KM = 1737;

/**
 * Earth distance profile over mission time.
 * t: 0..1 normalized
 */
function earthDistProfile(t) {
  if (t < 0.42) {
    // Outbound: smooth acceleration from LEO to near-lunar distance
    const p = t / 0.42;
    return R_EARTH_KM + (MOON_ORBIT_KM - R_EARTH_KM) * smoothStep(p);
  }
  if (t < 0.50) {
    // Flyby: stay near lunar distance, slight variation
    const p = (t - 0.42) / 0.08;
    return MOON_ORBIT_KM - 15000 * Math.sin(p * Math.PI);
  }
  // Return: decelerate back to Earth
  const p = (t - 0.50) / 0.50;
  return MOON_ORBIT_KM - (MOON_ORBIT_KM - R_EARTH_KM) * smoothStep(p);
}

/**
 * Moon distance profile — derived to create lateral offset in trilateration.
 *
 * For a free-return trajectory, the spacecraft doesn't stay on the
 * Earth-Moon line. We model this by making moon_dist slightly different
 * from |MOON_ORBIT - earth_dist|, creating a triangle with non-zero height.
 */
function moonDistProfile(earthDist, t) {
  // Baseline: if on the line, moonDist = |moonOrbit - earthDist|
  const onLine = Math.abs(MOON_ORBIT_KM - earthDist);

  if (t < 0.42) {
    // Outbound: spacecraft curves above the line
    const p = t / 0.42;
    // Add offset that creates ~20,000 km lateral displacement at midpoint
    const offset = 8000 * Math.sin(p * Math.PI);
    return Math.sqrt(onLine * onLine + offset * offset);
  }
  if (t < 0.50) {
    // Flyby: close to Moon
    const p = (t - 0.42) / 0.08;
    return MOON_RADIUS_KM + FLYBY_ALT_KM + 3000 * (1 - Math.sin(p * Math.PI));
  }
  // Return: curves below the line (opposite side)
  const p = (t - 0.50) / 0.50;
  const offset = 10000 * Math.sin(p * Math.PI);
  return Math.sqrt(onLine * onLine + offset * offset);
}

function smoothStep(t) {
  return t * t * (3 - 2 * t);
}

/**
 * Generate reference trajectory as Cartesian3 positions.
 * Uses the same trilaterateSpacecraft() as the live Orion position.
 */
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

/**
 * Generate Moon orbit ring.
 */
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

/**
 * Estimate launch timestamp from MET string.
 */
export function estimateLaunchTime(telemetryPoint) {
  if (!telemetryPoint) return null;
  const met = telemetryPoint.met;
  if (!met) return null;

  let totalSeconds = 0;

  const matchNew = met.match(/T\+(\d+)d\s+(\d+):(\d+):(\d+)/);
  if (matchNew) {
    totalSeconds =
      parseInt(matchNew[1]) * 86400 +
      parseInt(matchNew[2]) * 3600 +
      parseInt(matchNew[3]) * 60 +
      parseInt(matchNew[4]);
  } else {
    const matchOld = met.match(/(\d+):(\d+):(\d+):(\d+)/);
    if (matchOld) {
      totalSeconds =
        parseInt(matchOld[1]) * 86400 +
        parseInt(matchOld[2]) * 3600 +
        parseInt(matchOld[3]) * 60 +
        parseInt(matchOld[4]);
    }
  }

  if (totalSeconds === 0) return null;
  return telemetryPoint.timestamp - totalSeconds;
}
