/**
 * Selene-Insight — Artemis II Digital Twin
 *
 * Full mission animation with:
 * - Orion + Moon animated from JPL Horizons ECI vectors
 * - Sun lighting on Earth
 * - Mission event markers
 * - Telemetry charts
 */

import {
  Viewer,
  Cartesian3,
  Cartesian2,
  Color,
  Ion,
  LabelStyle,
  VerticalOrigin,
  HorizontalOrigin,
  NearFarScalar,
  SceneMode,
  JulianDate,
  ClockRange,
  ClockStep,
  SampledPositionProperty,
  LagrangePolynomialApproximation,
  PathGraphics,
  PolylineDashMaterialProperty,
  VelocityOrientationProperty,
  HeadingPitchRoll,
  Transforms,
  DirectionalLight,
  SunLight,
} from "cesium";
import "cesium/Build/Cesium/Widgets/widgets.css";

import { eciToCartesian3 } from "./lib/orbit.js";
import { generateMoonOrbit } from "./lib/referenceTrajectory.js";
import { buildColorSegments } from "./lib/colorScale.js";

// ── Config ──
Ion.defaultAccessToken =
  "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJlYWE1OWUxNy1mMWZiLTQzYjYtYTQ0OS1kMWFjYmFkNjc5YzciLCJpZCI6NTc1ODcsImlhdCI6MTYyNzg0NTE4Mn0.XcKpgANiY19MC4bdFUXMVEBToBmqS8kuYpUlxJHYZxk";

const MOON_RADIUS_M = 1.737e6;
const LABEL_FONT = "bold 22px monospace";

// Mission events (timestamps from Horizons analysis)
const MISSION_EVENTS = [
  { name: "LAUNCH", ts: 1743559200, icon: "🚀", color: "#00ff88" },
  { name: "TLI BURN", ts: 1743629400, icon: "🔥", color: "#ff8800" },
  { name: "LUNAR CLOSEST", ts: 1743980400, icon: "🌙", color: "#cccccc" },
  { name: "RETURN BEGINS", ts: 1744029000, icon: "↩", color: "#ffaa00" },
  { name: "PREDICTION END", ts: 1744328400, icon: "⏹", color: "#666666" },
];

// ── State ──
let alertBuffer = [];
let orionDataCache = [];

// ── Viewer ──
const viewer = new Viewer("cesium-container", {
  timeline: true,
  animation: true,
  homeButton: false,
  geocoder: false,
  sceneModePicker: false,
  baseLayerPicker: false,
  navigationHelpButton: false,
  fullscreenButton: false,
  infoBox: true,
  selectionIndicator: false,
  sceneMode: SceneMode.SCENE3D,
  skyBox: false,
  skyAtmosphere: false,
});
viewer.scene.backgroundColor = Color.BLACK;

// Sun lighting
viewer.scene.globe.enableLighting = true;

viewer.camera.setView({
  destination: Cartesian3.fromDegrees(20, 30, 800_000_000),
});

// ── Static Entities ──

const nowTs = Date.now() / 1000;

// Moon orbit ring
viewer.entities.add({
  name: "Moon Orbit",
  polyline: {
    positions: generateMoonOrbit(nowTs, Cartesian3),
    width: 1,
    material: Color.fromCssColorString("#444444").withAlpha(0.3),
  },
});

// Earth (slight exaggeration)
const EARTH_VIS_R = 6371000 * 1.2;
viewer.entities.add({
  name: "Earth",
  position: Cartesian3.fromDegrees(0, 0, 0),
  ellipsoid: {
    radii: new Cartesian3(EARTH_VIS_R, EARTH_VIS_R, EARTH_VIS_R),
    material: Color.fromCssColorString("#2244aa").withAlpha(0.8),
  },
  label: {
    text: "EARTH",
    font: LABEL_FONT,
    fillColor: Color.fromCssColorString("#4488ff"),
    style: LabelStyle.FILL_AND_OUTLINE,
    outlineColor: Color.BLACK,
    outlineWidth: 4,
    verticalOrigin: VerticalOrigin.BOTTOM,
    pixelOffset: new Cartesian2(0, -16),
    scaleByDistance: new NearFarScalar(5e5, 1.6, 1e9, 0.6),
  },
});

// ── Animated Entities ──

// Orion position
const orionPosition = new SampledPositionProperty();
orionPosition.setInterpolationOptions({
  interpolationDegree: 3,
  interpolationAlgorithm: LagrangePolynomialApproximation,
});

// Orion entity — oriented along velocity vector
const orionEntity = viewer.entities.add({
  name: "Orion",
  position: orionPosition,
  orientation: new VelocityOrientationProperty(orionPosition),
  point: { pixelSize: 14, color: Color.CYAN, outlineColor: Color.WHITE, outlineWidth: 2 },
  label: {
    text: "ORION ▸",
    font: LABEL_FONT,
    fillColor: Color.CYAN,
    style: LabelStyle.FILL_AND_OUTLINE,
    outlineColor: Color.BLACK,
    outlineWidth: 4,
    verticalOrigin: VerticalOrigin.BOTTOM,
    pixelOffset: new Cartesian2(0, -28),
    scaleByDistance: new NearFarScalar(5e5, 1.6, 1e9, 0.6),
  },
  // No PathGraphics trail — we use color-coded polyline segments instead
});

// Orion future path (dashed)
viewer.entities.add({
  name: "Predicted Path",
  position: orionPosition,
  path: new PathGraphics({
    leadTime: 86400 * 12,
    trailTime: 0,
    width: 2,
    material: new PolylineDashMaterialProperty({
      color: Color.fromCssColorString("#00ccff").withAlpha(0.25),
      dashLength: 16,
    }),
  }),
});

// Moon (animated)
const moonPosition = new SampledPositionProperty();
moonPosition.setInterpolationOptions({
  interpolationDegree: 3,
  interpolationAlgorithm: LagrangePolynomialApproximation,
});

viewer.entities.add({
  name: "Moon",
  position: moonPosition,
  ellipsoid: {
    radii: new Cartesian3(MOON_RADIUS_M, MOON_RADIUS_M, MOON_RADIUS_M),
    material: Color.fromCssColorString("#cccccc").withAlpha(0.9),
  },
  label: {
    text: "MOON",
    font: LABEL_FONT,
    fillColor: Color.fromCssColorString("#cccccc"),
    style: LabelStyle.FILL_AND_OUTLINE,
    outlineColor: Color.BLACK,
    outlineWidth: 4,
    verticalOrigin: VerticalOrigin.BOTTOM,
    pixelOffset: new Cartesian2(0, -28),
    scaleByDistance: new NearFarScalar(5e5, 1.6, 1e9, 0.6),
  },
});

// ── DOM refs ──
const dom = {
  met: document.getElementById("t-met"),
  phase: document.getElementById("t-phase"),
  velocity: document.getElementById("t-velocity"),
  earth: document.getElementById("t-earth"),
  moon: document.getElementById("t-moon"),
  source: document.getElementById("t-source"),
  count: document.getElementById("t-count"),
  grade: document.getElementById("v-grade"),
  confidence: document.getElementById("v-confidence"),
  vVel: document.getElementById("v-vel"),
  vEarth: document.getElementById("v-earth"),
  vMoon: document.getElementById("v-moon"),
  vCount: document.getElementById("v-count"),
  alertList: document.getElementById("alert-list"),
  connStatus: document.getElementById("connection-status"),
};

// ── Load full mission ──

fetch("/api/telemetry/history")
  .then((r) => r.json())
  .then((d) => {
    if (!d.orion || !d.orion.length) return;
    orionDataCache = d.orion;

    // Load Orion samples
    for (const p of d.orion) {
      const jd = JulianDate.fromDate(new Date(p.timestamp * 1000));
      orionPosition.addSample(jd, eciToCartesian3(p.pos_km, Cartesian3));
    }

    // Load Moon samples
    if (d.moon) {
      for (const p of d.moon) {
        const jd = JulianDate.fromDate(new Date(p.timestamp * 1000));
        moonPosition.addSample(jd, eciToCartesian3(p.pos_km, Cartesian3));
      }
    }

    // Set clock
    const startJd = JulianDate.fromDate(new Date(d.orion[0].timestamp * 1000));
    const stopJd = JulianDate.fromDate(new Date(d.orion[d.orion.length - 1].timestamp * 1000));
    const nowJd = JulianDate.fromDate(new Date());

    viewer.clock.startTime = startJd.clone();
    viewer.clock.stopTime = stopJd.clone();
    viewer.clock.currentTime = nowJd.clone();
    viewer.clock.clockRange = ClockRange.LOOP_STOP;
    viewer.clock.clockStep = ClockStep.SYSTEM_CLOCK_MULTIPLIER;
    viewer.clock.multiplier = 1;
    viewer.timeline.zoomTo(startJd, stopJd);

    // Add mission event markers
    addMissionEventMarkers(d.orion, d.moon);

    // Add velocity color-coded trajectory segments
    const segments = buildColorSegments(d.orion, eciToCartesian3, Cartesian3);
    for (const seg of segments) {
      viewer.entities.add({
        polyline: {
          positions: seg.positions,
          width: 4,
          material: seg.color,
        },
      });
    }

    // Tick handler for telemetry panel + chart
    viewer.clock.onTick.addEventListener((clock) => {
      updateFromClock(clock.currentTime);
    });

    console.log(`[MISSION] ${d.orion.length} Orion + ${d.moon?.length || 0} Moon points`);
  })
  .catch((e) => console.warn("[MISSION]", e));

// ── Mission Event Markers ──

function addMissionEventMarkers(orionData, moonData) {
  for (const evt of MISSION_EVENTS) {
    // Find closest Orion point to this event time
    let best = null;
    let bestDt = Infinity;
    for (const p of orionData) {
      const dt = Math.abs(p.timestamp - evt.ts);
      if (dt < bestDt) { bestDt = dt; best = p; }
    }
    if (!best || !best.pos_km) continue;

    const pos = eciToCartesian3(best.pos_km, Cartesian3);

    viewer.entities.add({
      name: evt.name,
      position: pos,
      point: {
        pixelSize: 8,
        color: Color.fromCssColorString(evt.color),
        outlineColor: Color.BLACK,
        outlineWidth: 1,
      },
      label: {
        text: `${evt.icon} ${evt.name}`,
        font: "bold 13px monospace",
        fillColor: Color.fromCssColorString(evt.color),
        style: LabelStyle.FILL_AND_OUTLINE,
        outlineColor: Color.BLACK,
        outlineWidth: 3,
        verticalOrigin: VerticalOrigin.TOP,
        horizontalOrigin: HorizontalOrigin.LEFT,
        pixelOffset: new Cartesian2(10, 0),
        scaleByDistance: new NearFarScalar(1e6, 1.0, 8e8, 0.4),
      },
    });
  }
}

// ── Telemetry Panel from Clock ──

let lastUpdateSec = 0;
function updateFromClock(currentTime) {
  const nowSec = JulianDate.toDate(currentTime).getTime() / 1000;
  if (Math.abs(nowSec - lastUpdateSec) < 0.4) return;
  lastUpdateSec = nowSec;

  if (!orionDataCache.length) return;

  // Find closest data point
  let best = null;
  let bestDt = Infinity;
  for (const p of orionDataCache) {
    const dt = Math.abs(p.timestamp - nowSec);
    if (dt < bestDt) { bestDt = dt; best = p; }
  }
  if (!best) return;

  const [x, y, z] = best.pos_km;
  const earthDist = Math.sqrt(x * x + y * y + z * z);
  const vel = best.vel_kms
    ? Math.sqrt(best.vel_kms[0] ** 2 + best.vel_kms[1] ** 2 + best.vel_kms[2] ** 2)
    : 0;

  const elapsed = nowSec - orionDataCache[0].timestamp;
  const days = Math.floor(elapsed / 86400);
  const hrs = Math.floor((elapsed % 86400) / 3600);
  const mins = Math.floor((elapsed % 3600) / 60);
  const secs = Math.floor(elapsed % 60);

  const isFuture = nowSec > Date.now() / 1000;

  // Phase detection
  let phase = "Outbound Coast";
  if (earthDist < 20000) phase = "Earth Orbit";
  else if (earthDist < 50000) phase = elapsed < 86400 * 4 ? "TLI Phase" : "Re-entry Approach";
  else if (earthDist > 350000) phase = "Lunar Vicinity";
  else if (elapsed > 86400 * 5) phase = "Return Coast";
  if (isFuture) phase = `${phase} (Predicted)`;

  dom.met.textContent = `T+${days}d ${pad(hrs)}:${pad(mins)}:${pad(secs)}`;
  dom.phase.textContent = phase;
  dom.velocity.textContent = vel ? `${vel.toFixed(3)} km/s` : "--";
  dom.earth.textContent = `${Math.round(earthDist).toLocaleString()} km`;
  dom.source.textContent = isFuture ? "prediction" : "jpl_horizons";
  dom.count.textContent = orionDataCache.length;

  // Update velocity chart bar
  updateMiniChart(vel, earthDist);
}

function pad(n) { return String(n).padStart(2, "0"); }

// ── Mini velocity/distance bars ──

function updateMiniChart(vel, earthDist) {
  const velBar = document.getElementById("vel-bar");
  const distBar = document.getElementById("dist-bar");
  if (velBar) velBar.style.width = `${Math.min(100, (vel / 12) * 100)}%`;
  if (distBar) distBar.style.width = `${Math.min(100, (earthDist / 450000) * 100)}%`;
}

// ── Alerts & Validation ──

fetch("/api/alerts/latest?n=10").then((r) => r.json())
  .then((d) => { if (d.data) d.data.reverse().forEach(addAlert); }).catch(() => {});

fetch("/api/validation/latest").then((r) => r.json())
  .then((d) => {
    if (d.recent && d.recent.length) updateValidation(d.recent[d.recent.length - 1]);
    if (d.stats) dom.vCount.textContent = d.stats.total_validations || 0;
  }).catch(() => {});

function updateValidation(data) {
  const grade = data.grade || "--";
  dom.grade.textContent = grade.toUpperCase();
  dom.grade.className = `grade-badge grade-${grade}`;
  dom.confidence.textContent = data.confidence != null ? `${(data.confidence * 100).toFixed(1)}%` : "--";
  const dev = data.deviations || {};
  dom.vVel.textContent = dev.velocity_pct != null ? `${dev.velocity_pct.toFixed(2)}%` : "--";
  dom.vEarth.textContent = dev.earth_dist_pct != null ? `${dev.earth_dist_pct.toFixed(2)}%` : "--";
  dom.vMoon.textContent = dev.moon_dist_pct != null ? `${dev.moon_dist_pct.toFixed(2)}%` : "--";
}

function addAlert(alert) {
  alertBuffer.push(alert);
  if (alertBuffer.length > 20) alertBuffer = alertBuffer.slice(-15);
  const el = document.createElement("div");
  el.className = "alert-item";
  const type = (alert.alert_type || "UNKNOWN").toUpperCase().replace("_", " ");
  el.innerHTML = `<div class="alert-type type-${alert.alert_type || ""}">${type}</div>
    <div class="alert-detail">${alert.details || ""}</div>`;
  dom.alertList.prepend(el);
  while (dom.alertList.children.length > 6) dom.alertList.removeChild(dom.alertList.lastChild);
}

// ── WebSocket ──
let validationCount = 0;
function connectWs() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${protocol}//${window.location.host}/ws/telemetry`);
  ws.onopen = () => { dom.connStatus.textContent = "LIVE"; dom.connStatus.className = "status-live"; };
  ws.onmessage = (evt) => {
    try {
      const msg = JSON.parse(evt.data);
      if (msg.type === "alert") addAlert(msg.data);
      else if (msg.type === "validation") { updateValidation(msg.data); dom.vCount.textContent = ++validationCount; }
    } catch {}
  };
  ws.onclose = () => { dom.connStatus.textContent = "OFFLINE"; dom.connStatus.className = "status-disconnected"; setTimeout(connectWs, 3000); };
  ws.onerror = () => ws.close();
}
connectWs();
