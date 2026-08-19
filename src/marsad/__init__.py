"""MARSAD 813 - hyperspectral harmful-algal-bloom early warning.

Arab 813 Space Hackathon 2026 entry. Three-stage pipeline:
Stage 1 shallow-water spectral correction, Stage 2 bloom detection &
speciation, Stage 3 drift & risk forecast per desalination intake.

v0.2 wraps that pipeline in the evidence layer that makes its claims
falsifiable: published operational baselines to compare against
(:mod:`marsad.baselines`), real multispectral band tables plus sub-pixel
mixing (:mod:`marsad.sensors`), a benchmark whose sensor ablation runs on
both the spectral and the spatial axis (:mod:`marsad.benchmark`),
per-prediction uncertainty with calibration and an analyst review queue
(:mod:`marsad.uncertainty`), a documented-event lead-time harness
(:mod:`marsad.hindcast`), and an alert feed for delivery
(:mod:`marsad.alerts`).

SCIENTIFIC HONESTY (binding, see docs/CONTRACTS-V2.md): :mod:`marsad.synth`
is our own physics-based forward model, so every benchmark number produced
by this package is a self-consistency check against a simulation, consistent
with the Case-2 water literature, and never independent validation on real
Gulf scenes. Independent validation is the archived-scene hindcast on
GLORIA / PACE / 813 data, and it has not been done yet.
"""

__version__ = "0.2.0"
