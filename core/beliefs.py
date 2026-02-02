import belief

def apply_continuous_decay(state, dt: float):
    state.stream_conf = belief.clamp01(state.stream_conf - belief.stream_dec(dt))
    state.map_conf = belief.clamp01(state.map_conf - belief.map_dec(dt))

def apply_stream_evidence(state, stream_status: int):
    state.stream_conf = belief.clamp01(state.stream_conf + belief.stream_inc(stream_status))

def apply_scan_boost(state):
    state.map_conf = belief.clamp01(state.map_conf + belief.map_scan_boost())
