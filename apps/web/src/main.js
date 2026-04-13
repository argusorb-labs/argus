/**
 * Selene-Insight — Starlink Constellation 3D Tracker
 *
 * Renders 10,000+ Starlink satellites on a CesiumJS globe
 * using PointPrimitiveCollection for GPU-accelerated performance.
 */

import {
  Viewer,
  Cartesian3,
  Color,
  Ion,
  SceneMode,
  PointPrimitiveCollection,
  SkyBox,
  ScreenSpaceEventType,
  ScreenSpaceEventHandler,
  defined,
} from "cesium";
import "cesium/Build/Cesium/Widgets/widgets.css";
import { createStarfieldSkyboxSources } from "./lib/starfield.js";

Ion.defaultAccessToken =
  "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJlYWE1OWUxNy1mMWZiLTQzYjYtYTQ0OS1kMWFjYmFkNjc5YzciLCJpZCI6NTc1ODcsImlhdCI6MTYyNzg0NTE4Mn0.XcKpgANiY19MC4bdFUXMVEBToBmqS8kuYpUlxJHYZxk";

// ── Shell color mapping ──
function shellColor(alt_km) {
  if (alt_km < 300) return Color.RED.withAlpha(0.8);        // deorbiting
  if (alt_km < 380) return Color.ORANGE.withAlpha(0.8);     // low shells
  if (alt_km < 460) return Color.YELLOW.withAlpha(0.7);     // mid shells
  if (alt_km < 520) return Color.CYAN.withAlpha(0.7);       // operational v2
  if (alt_km < 560) return Color.fromCssColorString("#4488ff").withAlpha(0.7); // v1 550
  return Color.fromCssColorString("#aa44ff").withAlpha(0.7); // high shells
}

// ── Viewer ──
const viewer = new Viewer("cesium-container", {
  timeline: false,
  animation: false,
  homeButton: false,
  geocoder: false,
  sceneModePicker: false,
  baseLayerPicker: false,
  navigationHelpButton: false,
  fullscreenButton: false,
  infoBox: false,
  selectionIndicator: false,
  sceneMode: SceneMode.SCENE3D,
  skyAtmosphere: false,
});

try {
  viewer.scene.skyBox = new SkyBox({
    sources: createStarfieldSkyboxSources(2048, 4000),
  });
} catch (e) { /* fallback to default */ }

viewer.scene.globe.enableLighting = true;
viewer.resolutionScale = window.devicePixelRatio || 1;

// Start zoomed out to see the full constellation
viewer.camera.setView({
  destination: Cartesian3.fromDegrees(0, 20, 20_000_000),
});

// ── Point Collection for satellites ──
const pointCollection = viewer.scene.primitives.add(new PointPrimitiveCollection());
const satelliteMap = new Map(); // norad_id → { point, data }

// ── DOM refs ──
const dom = {
  satCount: document.getElementById("sat-count"),
  shellList: document.getElementById("shell-list"),
  satDetail: document.getElementById("sat-detail"),
  anomalyList: document.getElementById("anomaly-list"),
  connStatus: document.getElementById("connection-status"),
};

// ── Update constellation ──
function updateConstellation(satellites) {
  const shells = {};

  for (const sat of satellites) {
    const pos = Cartesian3.fromDegrees(sat.lon, sat.lat, sat.alt_km * 1000);
    const color = shellColor(sat.alt_km);
    const key = sat.norad_id;

    shells[sat.shell_km] = (shells[sat.shell_km] || 0) + 1;

    if (satelliteMap.has(key)) {
      const entry = satelliteMap.get(key);
      entry.point.position = pos;
      entry.point.color = color;
      entry.data = sat;
    } else {
      const point = pointCollection.add({
        position: pos,
        color: color,
        pixelSize: 2.5,
      });
      point._noradId = key;
      satelliteMap.set(key, { point, data: sat });
    }
  }

  dom.satCount.textContent = satellites.length.toLocaleString();
  renderShells(shells);
}

function renderShells(shells) {
  const sorted = Object.entries(shells).sort((a, b) => b[1] - a[1]);
  dom.shellList.innerHTML = sorted.slice(0, 8).map(([km, count]) => {
    const c = shellColor(parseFloat(km));
    const hex = c.toCssHexString();
    return `<div class="shell-row">
      <span><span class="shell-dot" style="background:${hex}"></span><span class="shell-name">${km} km</span></span>
      <span class="shell-count">${count}</span>
    </div>`;
  }).join("");
}

// ── Click to select satellite ──
const handler = new ScreenSpaceEventHandler(viewer.scene.canvas);
handler.setInputAction((click) => {
  const picked = viewer.scene.pick(click.position);
  if (defined(picked) && picked.primitive && picked.primitive._noradId) {
    const noradId = picked.primitive._noradId;
    selectSatellite(noradId);
  }
}, ScreenSpaceEventType.LEFT_CLICK);

async function selectSatellite(noradId) {
  const entry = satelliteMap.get(noradId);
  if (!entry) return;

  // Highlight selected
  for (const [, e] of satelliteMap) e.point.pixelSize = 2.5;
  entry.point.pixelSize = 8;
  entry.point.color = Color.WHITE;

  const sat = entry.data;
  dom.satDetail.innerHTML = `
    <div class="sat-name">${sat.name || `NORAD ${sat.norad_id}`}</div>
    <div class="sat-row"><span class="sat-label">NORAD ID</span><span class="sat-value">${sat.norad_id}</span></div>
    <div class="sat-row"><span class="sat-label">ALTITUDE</span><span class="sat-value">${sat.alt_km.toFixed(1)} km</span></div>
    <div class="sat-row"><span class="sat-label">SHELL</span><span class="sat-value">${sat.shell_km} km</span></div>
    <div class="sat-row"><span class="sat-label">LAT</span><span class="sat-value">${sat.lat.toFixed(2)}°</span></div>
    <div class="sat-row"><span class="sat-label">LON</span><span class="sat-value">${sat.lon.toFixed(2)}°</span></div>
    <div class="sat-row"><span class="sat-label">STATUS</span><span class="sat-value">${sat.status}</span></div>
  `;

  // Fetch detailed info from API
  try {
    const r = await fetch(`/api/starlink/satellite/${noradId}`);
    const d = await r.json();
    if (d.satellite && d.tle_count) {
      dom.satDetail.innerHTML += `
        <div class="sat-row"><span class="sat-label">TLE RECORDS</span><span class="sat-value">${d.tle_count}</span></div>
      `;
    }
  } catch { /* ignore */ }
}

// ── Anomalies ──
function addAnomaly(a) {
  const name = satelliteMap.get(a.norad_id)?.data?.name || `NORAD ${a.norad_id}`;
  const el = document.createElement("div");
  el.className = "anomaly-item";
  el.innerHTML = `<div class="anom-type">${(a.anomaly_type || "").toUpperCase().replace("_", " ")}</div>
    <div class="anom-detail">${name}: ${a.details || ""}</div>`;
  if (dom.anomalyList.querySelector(".dim")) dom.anomalyList.innerHTML = "";
  dom.anomalyList.prepend(el);
  while (dom.anomalyList.children.length > 10) dom.anomalyList.removeChild(dom.anomalyList.lastChild);

  // Flash the satellite red
  const entry = satelliteMap.get(a.norad_id);
  if (entry) {
    entry.point.color = Color.RED;
    entry.point.pixelSize = 6;
  }
}

// ── Fetch initial data ──
fetch("/api/starlink/constellation")
  .then((r) => r.json())
  .then((d) => {
    if (d.satellites) updateConstellation(d.satellites);
  })
  .catch((e) => console.warn("Failed to load constellation:", e));

fetch("/api/starlink/anomalies?limit=10")
  .then((r) => r.json())
  .then((d) => {
    if (d.anomalies) d.anomalies.forEach(addAnomaly);
  })
  .catch(() => {});

// ── WebSocket ──
function connectWs() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${protocol}//${window.location.host}/ws/telemetry`);

  ws.onopen = () => {
    dom.connStatus.textContent = "LIVE";
    dom.connStatus.className = "status-live";
  };

  ws.onmessage = (evt) => {
    try {
      const msg = JSON.parse(evt.data);
      if (msg.type === "positions") {
        // Positions broadcast is lightweight (just count), re-fetch full data
        fetch("/api/starlink/constellation")
          .then((r) => r.json())
          .then((d) => { if (d.satellites) updateConstellation(d.satellites); });
      } else if (msg.type === "anomaly") {
        addAnomaly(msg.data);
      }
    } catch { /* ignore */ }
  };

  ws.onclose = () => {
    dom.connStatus.textContent = "OFFLINE";
    dom.connStatus.className = "status-disconnected";
    setTimeout(connectWs, 5000);
  };
  ws.onerror = () => ws.close();
}
connectWs();
