# belief.py

# STREAM confidence: "is the sensor feed trustworthy right now?"
STREAM_INC = 0.01          # gain per successful saved frame (1 fps -> ~0.01/sec)
STREAM_DECAY_RATE = 0.001  # per second

# MAP confidence: "is my world model fresh?"
MAP_DECAY_RATE = 0.002     # per second (decays even if stream is healthy)
MAP_SCAN_BOOST = 0.25      # boost when an active scan occurs (manual trigger for now)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def stream_inc(stream_status: int) -> float:
    # Return a number ALWAYS
    return STREAM_INC if stream_status == 1 else 0.0


def stream_dec(dt: float) -> float:
    # dt is seconds
    return STREAM_DECAY_RATE * float(dt)


def map_dec(dt: float) -> float:
    return MAP_DECAY_RATE * float(dt)


def map_scan_boost() -> float:
    return MAP_SCAN_BOOST
