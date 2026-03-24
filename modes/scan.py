from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
import time
import numpy as np

from core import logger
from core.perception.features import extract_features, is_observation_valid


@dataclass
class ScanRuntime:
    active: bool = False
    phase: str = "rotate"      # rotate | pause
    step_idx: int = 0          # 0..3
    episode: int = 0           # 0..cfg.scan_max_episodes-1
    phase_t0: float = 0.0
    pause_sampled: bool = False
    stored_repr: np.ndarray | None = None
    rotation_started: bool = False


def enter(state, now_t, cfg):
    state.scan_rt = ScanRuntime(
        active=True,
        phase="rotate",
        step_idx=0,
        episode=0,
        phase_t0=time.monotonic(),
        pause_sampled=False,
        stored_repr=None,
        rotation_started=False,
    )
    logger.log_event(state.log_path, datetime.now(), "scan_enter", state, notes="entered scan mode")


def exit(state, now_t, cfg):
    logger.log_event(state.log_path, datetime.now(), "scan_exit", state, notes="leaving scan mode")
    state.scan_rt = None


def _current_heading_bin(state) -> int:
    # If heading_deg is not present yet, assume 0
    heading = getattr(state, "heading_deg", 0.0)
    return int(round((heading % 360.0) / 90.0)) % 4


def _rotation_duration_s(cfg, degrees: float = 90.0) -> float:
    return min(abs(degrees) * cfg.turn_seconds_per_degree, cfg.max_turn_time_s)


def step(state, now_t, cfg, rotate_fn=None, sample_frame_fn=None):
    """
    Returns:
        next_mode (str | None)
        - None => keep current mode
        - "observe" => request mode switch back to observe
    """
    rt = state.scan_rt
    if rt is None or not rt.active:
        return None

    now_mono = time.monotonic()

    # -------------------------
    # Phase A: ROTATE 90 deg
    # -------------------------
    if rt.phase == "rotate":
        if not rt.rotation_started:
            # Start rotation once at phase entry.
            # rotate_fn is expected to be blocking in your current main.py.
            # So this design assumes turn time is short and we stamp phase completion after call.
            if rotate_fn is not None and state.motors_enabled:
                rotate_fn(90.0)

            rt.rotation_started = True
            rt.phase_t0 = now_mono

            logger.log_event(
                state.log_path,
                datetime.now(),
                "scan_step_rotate_start",
                state,
                notes=f"episode={rt.episode+1} step={rt.step_idx+1}/4"
            )

            # Since rotate_fn is blocking, immediately transition to pause.
            rt.phase = "pause"
            rt.phase_t0 = now_mono
            rt.pause_sampled = False
            rt.rotation_started = False

            logger.log_event(
                state.log_path,
                datetime.now(),
                "scan_step_rotate_done",
                state,
                notes=f"episode={rt.episode+1} step={rt.step_idx+1}/4"
            )

        return None

    # -------------------------
    # Phase B: PAUSE & SAMPLE ONCE
    # -------------------------
    if rt.phase == "pause":
        if not rt.pause_sampled:
            ok, frame = sample_frame_fn() if sample_frame_fn is not None else (False, None)

            if ok and frame is not None:
                features, feature_vector = extract_features(frame, cfg)
                valid, reason, reliability, rel_label, rel_comps = is_observation_valid(
                    features,
                    feature_vector,
                    state.recent_feature_vectors,
                    cfg,
                )

                # Update history after reliability computation
                state.recent_feature_vectors.append(feature_vector.copy())
                if len(state.recent_feature_vectors) > cfg.recent_feature_window:
                    state.recent_feature_vectors.pop(0)

                ep = rt.episode + 1
                st = rt.step_idx + 1
                k = _current_heading_bin(state)
                M_k = state.memory_bins.get(k)
                novelty_k = None
                beta = 0.0
                mem_upd = 0

                if not valid:
                    logger.log_event(
                        state.log_path,
                        datetime.now(),
                        "scan_step_obs",
                        state,
                        notes=(
                            f"episode={ep} step={st}/4 "
                            f"valid=0 label={rel_label} reliability={reliability:.3f} "
                            f"reason={reason} "
                            f"r_sharp={rel_comps['r_sharp']:.2f} "
                            f"r_histo={rel_comps['r_histo']:.2f} "
                            f"r_temp={rel_comps['r_temp']:.2f} "
                            f"sharp_q={features['sharp_q']:.2f} "
                            f"edge_q={features['edge_q']:.2f} "
                            f"kp_q={features['kp_q']:.2f} "
                            f"mean={features['mean_intensity']:.1f} "
                            f"std={features['std_intensity']:.1f}"
                        )
                    )

                else:
                    # --------------------------------
                    # Init short-term model only from trusted frames
                    # --------------------------------
                    if rt.stored_repr is None:
                        if rel_label != "trusted":
                            logger.log_event(
                                state.log_path,
                                datetime.now(),
                                "scan_step_obs",
                                state,
                                notes=(
                                    f"episode={ep} step={st}/4 "
                                    f"valid=1 init=0 label={rel_label} reliability={reliability:.3f} "
                                    f"reason={reason} "
                                    f"sharp_q={features['sharp_q']:.2f} "
                                    f"edge_q={features['edge_q']:.2f} "
                                    f"kp_q={features['kp_q']:.2f} "
                                    f"mean={features['mean_intensity']:.1f} "
                                    f"std={features['std_intensity']:.1f}"
                                )
                            )
                        else:
                            rt.stored_repr = feature_vector.astype(np.float32, copy=False)

                            # Trusted init frame may initialize / update long-term memory
                            if M_k is None:
                                novelty_k = 1.0
                                state.memory_bins[k] = feature_vector.astype(np.float32, copy=False)
                                mem_upd = 1
                            else:
                                novelty_k = float(np.mean(np.abs(feature_vector - M_k)))
                                dd = float(getattr(cfg, "novelty_deadband", 0.0))
                                novelty_k = max(0.0, novelty_k - dd)

                                beta = novelty_k
                                if getattr(cfg, "memory_beta_floor", 0.0) > 0.0 and novelty_k > 0.0:
                                    beta = max(beta, float(cfg.memory_beta_floor))
                                beta = min(1.0, beta)

                                if state.map_conf >= cfg.memory_stability_thresh:
                                    state.memory_bins[k] = (
                                        (1.0 - beta) * M_k + beta * feature_vector
                                    ).astype(np.float32, copy=False)
                                    mem_upd = 1

                            logger.log_event(
                                state.log_path,
                                datetime.now(),
                                "scan_step_obs",
                                state,
                                notes=(
                                    f"episode={ep} step={st}/4 "
                                    f"valid=1 init=1 label={rel_label} reliability={reliability:.3f} "
                                    f"reason={reason} "
                                    f"sharp_q={features['sharp_q']:.2f} "
                                    f"edge_q={features['edge_q']:.2f} "
                                    f"kp_q={features['kp_q']:.2f} "
                                    f"novelty_k={novelty_k if novelty_k is not None else 'NA'} "
                                    f"beta={beta:.2f} mem_upd={mem_upd}"
                                )
                            )

                    # --------------------------------
                    # Normal prediction-error update
                    # --------------------------------
                    else:
                        e = float(np.mean(np.abs(feature_vector - rt.stored_repr)))
                        effective_e = reliability * e

                        if rel_label == "suspect":
                            effective_e *= cfg.suspect_influence

                        prev = state.map_conf
                        state.map_conf -= cfg.scan_k_drop * effective_e
                        state.map_conf += cfg.scan_k_gain * (1.0 - effective_e)
                        state.map_conf = max(0.0, min(1.0, state.map_conf))

                        alpha = cfg.scan_repr_alpha
                        if rel_label == "trusted":
                            rt.stored_repr = ((1.0 - alpha) * rt.stored_repr) + (alpha * feature_vector)
                            rt.stored_repr = rt.stored_repr.astype(np.float32, copy=False)

                        # Novelty vs long-term memory
                        if M_k is None:
                            novelty_k = 1.0
                            beta = 1.0
                        else:
                            novelty_k = float(np.mean(np.abs(feature_vector - M_k)))
                            dd = float(getattr(cfg, "novelty_deadband", 0.0))
                            novelty_k = max(0.0, novelty_k - dd)

                            beta = novelty_k
                            if getattr(cfg, "memory_beta_floor", 0.0) > 0.0 and novelty_k > 0.0:
                                beta = max(beta, float(cfg.memory_beta_floor))
                            beta = min(1.0, beta)

                        mem_upd = 0
                        if rel_label == "trusted":
                            if M_k is None:
                                state.memory_bins[k] = feature_vector.astype(np.float32, copy=False)
                                mem_upd = 1
                            elif state.map_conf >= cfg.memory_stability_thresh:
                                state.memory_bins[k] = (
                                    (1.0 - beta) * M_k + beta * feature_vector
                                ).astype(np.float32, copy=False)
                                mem_upd = 1

                        logger.log_event(
                            state.log_path,
                            datetime.now(),
                            "scan_step_obs",
                            state,
                            notes=(
                                f"episode={ep} step={st}/4 "
                                f"valid=1 label={rel_label} rel={reliability:.3f} "
                                f"reason={reason} "
                                f"sharp_q={features['sharp_q']:.2f} "
                                f"edge_q={features['edge_q']:.2f} "
                                f"kp_q={features['kp_q']:.2f} "
                                f"e={e:.3f} eff_e={effective_e:.3f} "
                                f"map_conf:{prev:.2f}->{state.map_conf:.2f} "
                                f"alpha={alpha:.2f} "
                                f"novelty_k={novelty_k if novelty_k is not None else 'NA'} "
                                f"beta={beta:.2f} mem_upd={mem_upd} "
                                f"r_sharp={rel_comps['r_sharp']:.2f} "
                                f"r_histo={rel_comps['r_histo']:.2f} "
                                f"r_temp={rel_comps['r_temp']:.2f}"
                            )
                        )

            else:
                logger.log_event(
                    state.log_path,
                    datetime.now(),
                    "scan_step_obs",
                    state,
                    notes="camera_read_failed"
                )

            rt.pause_sampled = True

        # end pause, advance scan machine
        if now_mono - rt.phase_t0 >= cfg.scan_pause_s:
            rt.step_idx += 1

            if rt.step_idx >= 4:
                rt.episode += 1
                rt.step_idx = 0
                rt.stored_repr = None

                bootstrap_needed_now = any(state.memory_bins.get(k) is None for k in range(4))

                logger.log_event(
                    state.log_path,
                    datetime.now(),
                    "scan_episode_done",
                    state,
                    notes=(
                        f"episode={rt.episode}/{cfg.scan_max_episodes} "
                        f"bootstrap_needed={int(bootstrap_needed_now)}"
                    )
                )

                if state.map_conf >= cfg.scan_exit_thresh and not bootstrap_needed_now:
                    return "observe"

                if rt.episode >= cfg.scan_max_episodes:
                    return "observe"

            rt.phase = "rotate"
            rt.phase_t0 = time.monotonic()
            rt.rotation_started = False

            # keep heading estimate if you use it elsewhere
            state.heading_deg = (getattr(state, "heading_deg", 0.0) + 90.0) % 360.0

            logger.log_event(
                state.log_path,
                datetime.now(),
                "scan_step_advance",
                state,
                notes=f"next episode={rt.episode+1} step={rt.step_idx+1}/4"
            )

    return None