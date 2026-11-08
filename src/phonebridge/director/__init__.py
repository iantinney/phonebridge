"""The session director: scrolls like the operator and reacts to what is on screen (PLAN §6, lane director).

`session.Director` runs one feed session per post: arrive → fast local cues (`cues`) decide skip/stay → dwell and
the leaving flick are drawn by smoothed bootstrap from the persona's pools (`dwell`, `flick`, `persona`) → small
habits (`habits`) → a background vision model proposes writes only when writes are allowed (`content_vlm`); every
write is the caller's and is verified. `realism` reports drift between the pools and the draws (a diagnostic, not
a gate). Calibration of the pools from the operator's recordings and exports lives in `phonebridge.calibrate`.
"""
