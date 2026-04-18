"""IMM-UKF Classifier — multi-model anomaly detection for ArgusOrb.

Runs an IMM-UKF over a satellite's TLE history. Three competing models:
  M0: station-keeping (small process noise, normal ops)
  M1: maneuver (large process noise on velocity, allows sudden Δv)
  M2: decay (large process noise on along-track, altitude trending down)

The posterior model probability P(Mk|data) at each TLE transition is
the anomaly label. When P(M1) spikes, the satellite likely maneuvered.
When P(M2) dominates, it's decaying uncontrolled.

Labels are written to the anomaly table with classified_by="imm_ukf_v1",
coexisting with rule_v1 labels. This is the A/B mechanism — the two
classifiers' opinions on the same events can be compared directly.

Usage:
    python -m services.brain.imm_classifier 44714
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

try:
    from services.brain.dynamics import propagate_state, tle_to_state
    from services.brain.ukf import UKF
    from services.brain.imm import IMM
    from services.telemetry.store import StarlinkStore
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from services.brain.dynamics import propagate_state, tle_to_state
    from services.brain.ukf import UKF
    from services.brain.imm import IMM
    from services.telemetry.store import StarlinkStore


# ── Model definitions ──

MODEL_NAMES = {0: "station_keeping", 1: "maneuver", 2: "decay"}

# Process noise covariance Q for each model.
# State = [x, y, z, vx, vy, vz] in meters and m/s.
# Q diagonal: [pos_noise², pos_noise², pos_noise², vel_noise², vel_noise², vel_noise²]

def _make_Q(pos_sigma_m: float, vel_sigma_ms: float) -> np.ndarray:
    """Build a diagonal process noise covariance."""
    return np.diag([
        pos_sigma_m**2, pos_sigma_m**2, pos_sigma_m**2,
        vel_sigma_ms**2, vel_sigma_ms**2, vel_sigma_ms**2,
    ])

# Base Q values at 550 km reference altitude.
# Scaled by (REF_ALT / actual_alt)² for altitude-adaptive behavior.
REF_ALT_KM = 550.0

# M0: station-keeping — small noise (elements drift ~100m/orbit, ~0.01 m/s)
Q_SK_BASE = _make_Q(pos_sigma_m=100.0, vel_sigma_ms=0.01)

# M1: maneuver — large velocity noise (allows ~1 m/s Δv between TLEs)
Q_MAN_BASE = _make_Q(pos_sigma_m=500.0, vel_sigma_ms=1.0)

# M2: decay — large position noise (orbit shrinking due to drag)
Q_DECAY_BASE = _make_Q(pos_sigma_m=2000.0, vel_sigma_ms=0.1)

# Observation noise R: TLE → sgp4 gives ~1 km position, ~1 m/s velocity
R_TLE = np.diag([
    1000.0**2, 1000.0**2, 1000.0**2,   # 1 km position noise
    1.0**2, 1.0**2, 1.0**2,             # 1 m/s velocity noise
])

# Initial covariance: start with TLE-level uncertainty
P0 = np.diag([
    2000.0**2, 2000.0**2, 2000.0**2,   # 2 km position
    2.0**2, 2.0**2, 2.0**2,             # 2 m/s velocity
])

# IMM transition matrix at reference altitude (550 km).
# At lower altitudes, decay transition probability increases.
T_MATRIX_BASE = np.array([
    [0.97, 0.015, 0.015],  # from station-keeping
    [0.10, 0.85,  0.05],   # from maneuver (short-lived, likely returns to SK)
    [0.02, 0.03,  0.95],   # from decay (tends to persist)
])


def _altitude_scale(alt_km: float) -> float:
    """Scaling factor for Q_decay: higher drag at lower altitude.

    At 550 km: scale = 1.0
    At 275 km: scale = 4.0 (drag ~100× stronger, Q needs to be wider)
    At 200 km: scale = 7.6
    Clamped to [1, 20] to avoid runaway.
    """
    if alt_km <= 0:
        return 20.0
    scale = (REF_ALT_KM / max(alt_km, 150.0)) ** 2
    return min(scale, 20.0)


def _altitude_adjusted_Qs(alt_km: float) -> list[np.ndarray]:
    """Return [Q_sk, Q_maneuver, Q_decay] scaled for current altitude."""
    s = _altitude_scale(alt_km)
    return [
        Q_SK_BASE,              # station-keeping Q doesn't scale
        Q_MAN_BASE,             # maneuver Q doesn't scale (Δv is Δv)
        Q_DECAY_BASE * s,       # decay Q scales with drag intensity
    ]


def _altitude_adjusted_priors(alt_km: float) -> np.ndarray:
    """Model priors that shift toward decay at low altitude.

    At 550 km:  [0.90, 0.05, 0.05]  — mostly station-keeping
    At 350 km:  [0.60, 0.05, 0.35]  — decay becomes plausible
    At 250 km:  [0.30, 0.05, 0.65]  — decay is dominant prior
    At 200 km:  [0.10, 0.05, 0.85]  — almost certainly decaying
    """
    if alt_km >= 500:
        return np.array([0.90, 0.05, 0.05])
    elif alt_km >= 350:
        # Linear interpolation 500→350: decay prior 0.05→0.35
        t = (500 - alt_km) / 150.0
        decay_prior = 0.05 + t * 0.30
        sk_prior = 1.0 - 0.05 - decay_prior
        return np.array([sk_prior, 0.05, decay_prior])
    elif alt_km >= 200:
        t = (350 - alt_km) / 150.0
        decay_prior = 0.35 + t * 0.50
        sk_prior = 1.0 - 0.05 - decay_prior
        return np.array([max(sk_prior, 0.05), 0.05, decay_prior])
    else:
        return np.array([0.05, 0.05, 0.90])


def _altitude_adjusted_T(alt_km: float) -> np.ndarray:
    """Transition matrix that increases decay persistence at low altitude."""
    T = T_MATRIX_BASE.copy()
    if alt_km < 350:
        # Increase P(sk → decay) and P(decay → decay) at low altitude
        boost = min((350 - alt_km) / 150.0, 1.0) * 0.10
        T[0, 2] += boost          # sk → decay
        T[0, 0] -= boost          # less persistence in sk
        T[2, 2] = min(T[2, 2] + boost * 0.5, 0.99)  # more decay persistence
        T[2, 0] = max(T[2, 0] - boost * 0.5, 0.005)
    return T


def _fx_wrapper(state: np.ndarray, dt: float, bstar: float = 0.0) -> np.ndarray:
    """State transition function for the UKF (single sigma point)."""
    result, success = propagate_state(state, dt, bstar=bstar)
    return result


def _batch_fx_wrapper(sigmas: np.ndarray, dt: float, bstar: float = 0.0):
    """Batch state transition: vectorized RK4 for all sigma points at once."""
    from services.brain.dynamics import propagate_batch_rk4
    return propagate_batch_rk4(sigmas, dt, bstar=bstar)


def _hx_identity(state: np.ndarray) -> np.ndarray:
    """Observation model: we observe the full state (TLE → sgp4 → state)."""
    return state


def create_imm(initial_state: np.ndarray, alt_km: float = REF_ALT_KM) -> IMM:
    """Create a fresh IMM-UKF instance with altitude-adaptive parameters.

    At high altitude (>500 km): standard Q, strong station-keeping prior.
    At low altitude (<350 km): inflated Q_decay, strong decay prior.
    This fixes the reentry detection gap where standard Q couldn't
    distinguish decay from station-keeping at 200 km.
    """
    Qs = _altitude_adjusted_Qs(alt_km)
    priors = _altitude_adjusted_priors(alt_km)
    T = _altitude_adjusted_T(alt_km)

    filters = []
    for i in range(3):
        ukf = UKF(
            n_state=6, n_obs=6,
            fx=_fx_wrapper, hx=_hx_identity,
            alpha=1e-2, beta=2.0, kappa=0.0,
        )
        ukf.x = initial_state.copy()
        ukf.P = P0.copy()
        ukf.Q = Qs[i]
        ukf.R = R_TLE.copy()
        filters.append(ukf)

    return IMM(
        filters=filters,
        model_probs=priors,
        transition_matrix=T,
    )


def classify_satellite_history(
    store: StarlinkStore,
    norad_id: int,
    max_history: int = 200,
) -> list[dict]:
    """Run IMM-UKF over a satellite's TLE history and return labels.

    Processes TLEs from oldest to newest. At each transition, records
    the model probabilities as an anomaly label.

    Returns list of label dicts (ready for store.insert_anomaly).
    """
    history = store.get_satellite_history(norad_id, limit=max_history)
    if len(history) < 2:
        return []

    # Process oldest → newest
    history.reverse()

    # Initialize from the first TLE
    first = history[0]
    state0 = tle_to_state(first.get("line1", ""), first.get("line2", ""))
    if state0 is None:
        return []

    # Estimate initial altitude for adaptive parameters
    from services.brain.dynamics import R_EARTH
    alt0_km = np.linalg.norm(state0[:3]) / 1000 - R_EARTH / 1000
    imm = create_imm(state0, alt_km=alt0_km)
    labels: list[dict] = []

    for i in range(1, len(history)):
        prev_tle = history[i - 1]
        curr_tle = history[i]

        # Time between TLEs
        dt_days = (curr_tle.get("epoch_jd", 0) - prev_tle.get("epoch_jd", 0))
        dt_seconds = dt_days * 86400.0
        if dt_seconds <= 0 or dt_seconds > 7 * 86400:
            # Skip backwards or very large gaps (>7 days)
            continue

        bstar = curr_tle.get("bstar") or prev_tle.get("bstar") or 0.0

        # Predict (using batch propagation for speed)
        fx_args = [(bstar,)] * 3
        try:
            imm.predict(dt=dt_seconds, fx_args_per_model=fx_args,
                        batch_fx=_batch_fx_wrapper)
        except Exception:
            continue

        # Observe: current TLE → state vector
        obs_state = tle_to_state(
            curr_tle.get("line1", ""), curr_tle.get("line2", "")
        )
        if obs_state is None:
            continue

        # Update
        try:
            imm.update(obs_state)
        except Exception:
            continue

        # Record label if a non-station-keeping model dominates
        probs = imm.model_probabilities
        best_model = imm.most_likely_model
        best_prob = probs[best_model]

        if best_model != 0 and best_prob > 0.3:
            # Non-station-keeping model has significant probability
            model_name = MODEL_NAMES[best_model]

            from services.brain.dynamics import R_EARTH
            alt_km = np.linalg.norm(imm.x[:3]) / 1000 - R_EARTH / 1000

            label = {
                "norad_id": norad_id,
                "anomaly_type": f"imm_{model_name}",
                "cause": "maneuver_candidate" if best_model == 1 else "natural_decay",
                "confidence": round(best_prob, 4),
                "classified_by": "imm_ukf_v1",
                "source_epoch_jd": curr_tle.get("epoch_jd"),
                "altitude_before_km": None,
                "altitude_after_km": round(alt_km, 1) if alt_km > 0 else None,
                "details": (
                    f"IMM-UKF: P(station_keeping)={probs[0]:.3f} "
                    f"P(maneuver)={probs[1]:.3f} "
                    f"P(decay)={probs[2]:.3f}"
                ),
            }
            labels.append(label)

    return labels


# ── CLI ──

def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m services.brain.imm_classifier",
        description="Run IMM-UKF classifier on a satellite's TLE history.",
    )
    parser.add_argument("norad_id", type=int, help="NORAD catalog ID")
    parser.add_argument("--db", help="Path to SQLite store")
    parser.add_argument("--write", action="store_true",
                        help="Write labels to the anomaly table")
    args = parser.parse_args(argv)

    db = args.db or os.environ.get("ARGUS_DB_PATH", "data/starlink.db")
    store = StarlinkStore(db)

    t0 = time.time()
    labels = classify_satellite_history(store, args.norad_id)
    elapsed = time.time() - t0

    print(f"IMM-UKF: {len(labels)} events for NORAD {args.norad_id} "
          f"in {elapsed:.2f}s")

    for label in labels:
        model_type = label["anomaly_type"].replace("imm_", "")
        conf = label["confidence"]
        details = label["details"]
        print(f"  [{model_type:>16s}] conf={conf:.3f}  {details}")

    if args.write and labels:
        now = time.time()
        written = 0
        for label in labels:
            label["detected_at"] = now
            if store.insert_anomaly(label):
                written += 1
        print(f"\nWrote {written} new labels to anomaly table "
              f"(classified_by=imm_ukf_v1)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
