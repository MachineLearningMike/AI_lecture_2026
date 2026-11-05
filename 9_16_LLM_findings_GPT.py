"""
9_16_LLM_findings_GPT.py

LLM performance as a multidimensional, time-dependent function: an analysis of
"The shelf life of LLM findings" (Queen & Zou, Stanford, November 2, 2026),
checked against the study's own data.

Companion to 9_15_LLM_findings_Claude.py, which presents the study itself. This deck
follows a written analysis of that study (GPT-assisted notes supplied for this
lecture): eight performance dimensions (A-H), capability failures vs behavioural
tendencies, a performance-vector framework, a critique of the method, and a
proposal for a longitudinal capability database. Each claim in the analysis is
matched to the study's evidence. Where the two disagree, the slide says so.

Slides:
  0. Preface                             14. Two classes of behaviour (live)
  1. The central idea                    15. A possible development trajectory
  2. Eight dimensions of performance     16. Performance vector P(M_t) (live)
  3-10. Dimensions A-H vs the evidence   17. Do dimensions improve at one rate?
 11. Where the analysis needs correcting 18. Model identity matters (heat map)
 12. Correlation is not causation        19. Trade-offs and patching
 13. Critical evaluation of the study    20. Dimensions the study did not cover
                                         21. A longitudinal capability DB (live)
                                         22. Conclusions

Part A adds what 9_15 does not have: per-model relative effects for all 50
claims x 20 models (taken from the study page), the analysis's dimension map
with claim evidence, the performance vector, per-dimension improvement rates,
and an in-memory SQLite version of the proposed longitudinal database.

Usage:
  python 9_16_LLM_findings_GPT.py            # slide deck
  python 9_16_LLM_findings_GPT.py --demo     # vectors, rates and DB queries in the terminal
  python 9_16_LLM_findings_GPT.py --map      # the dimension map with evidence

Requires 9_15_LLM_findings_Claude.py in the same folder (claim table + slide helpers).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import sqlite3
import sys
import textwrap
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np


def _load_base():
    """Import 9_15_LLM_findings_Claude.py (its name starts with a digit, so no plain import)."""
    name = "llm_findings_9_15"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name("9_15_LLM_findings_Claude.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


base = _load_base()
CLAIMS, CLAIM, AREA_SHORT, PANEL = base.CLAIMS, base.CLAIM, base.AREA_SHORT, base.PANEL
VERDICTS = base.VERDICTS


# =======================================================
# Part A: per-model data for all 50 claims
# =======================================================
MODELS = [m for m, _ in PANEL]
RELEASE = {m: date.fromisoformat(d) for m, d in PANEL}

# Relative effect e of each claim on each panel model (order = PANEL);
# None = not reliably scorable (refusals, unparseable answers).
PANEL_E_ALL = {
    "C1": [0.455, 0.565, None, 0.436, 2.03, 0.521, 0.47, 0.286, 0.708, 0.452, 0.173, 0.336, 0.2, 0.461, 0.622, 0.107, 0.18, 0.343, 0.238, 0.156],
    "C2": [0.902, 0.865, 0.183, 0.841, 1.341, 0.829, 1.134, 0.524, 0, 0, 0.037, 0, 0.244, 0.381, 0.693, 0, 0, 0.183, 0.902, 0.524],
    "C3": [1.163, 0.626, 0.838, 0.595, 0.439, 0.763, 0.27, 0.649, 0.595, 0.406, 0.189, 0.676, 0.517, 0.27, 0.371, 0.695, 0.243, 0.191, 0.135, 0.191],
    "C4": [0.649, 0.513, 0.919, 0.351, 0.513, 0.622, 0.513, 0.575, 0.892, 0.46, 0.365, 0.486, 0.351, 0.432, 0.903, 0.356, 0.486, 0.351, 0.836, 0.54],
    "C5": [1.283, 0.718, 1.095, 1.377, 0.53, 0.671, 0.906, 0.577, 0.671, -0.034, 0.577, 0.013, 0.295, 0.295, 0.013, 0.013, 0.248, -0.034, -0.034, -0.034],
    "C6": [0, 0.418, 0.026, 0, 0.209, 0.026, 0, 0, 0.026, 0.157, 0, 0, 0, 0, 0, 0, 0, 0.026, 0, 0],
    "C7": [0.321, -0.212, 0.446, -0.085, 0.319, 0.17, 0.34, 0.276, -0.106, 0.403, 0.087, -0.255, 0.17, -0.255, 0.395, 0.087, -0.086, -0.043, -0.149, 0.212],
    "C8": [0.394, 0.151, 0.061, 0.061, 0.182, 0.212, 0.333, 1.182, 0.182, 1.03, 0.092, 0.091, 0.03, 0.061, 0.639, 0.061, 0, 0.091, 0.091, 0.061],
    "C9": [1.315, 0.969, 1.12, 1.045, 0.94, 0.767, 1.072, 0.857, 1.169, 1.186, 0.621, 0.704, 0.479, 0.502, 0.765, 0.643, 0.725, 1.199, 0.618, 0.892],
    "C10": [1.054, 1.236, 1.182, 1.364, 1.254, 1.327, 1, 1.345, 1.345, 1.341, 1.307, 1.436, 1.127, 1.345, 1.322, 1.377, 1.109, 1.327, 1.254, 1.364],
    "C11": [1.233, 0.667, 0.869, 0.363, 0.571, 0.482, 0.739, 1.167, 0.433, 0.094, 0.373, 0.078, 0.198, 1.246, 0.087, 0.147, 0.309, 0.051, 0.025, 0],
    "C12": [0.902, -0.07, 0.003, 0.134, 0.124, 0.037, 0.142, -0.031, -0.105, 0.107, -0.042, 0.038, -0.041, 0.064, 0.041, -0.028, -0.049, 0.013, -0.024, 0.019],
    "C13": [1.183, 0.987, 0.586, 1.387, None, None, 1.562, 0.493, 1.221, 1.745, 0.6, 0.776, 0.662, 0.526, 0.511, 0.572, 0.593, 0.616, 0.533, 0.406],
    "C14": [0.579, 0.255, 0.026, 0.605, 0.053, 1.237, 0.824, 0.242, 0.421, 0.526, 1.223, None, 0.895, 1.21, 0.316, 1.184, 0.028, 0.184, None, 0.053],
    "C15": [0.72, 0.807, 0.303, 0.303, 1.183, 0.039, 0.038, -0.115, 0.872, 0.341, 0.269, 0.038, 0.152, 0.076, 0.166, 0.273, 0.19, 0.076, 0.076, 0.076],
    "C16": [0.463, 0.395, 0.212, 0.056, None, 0.106, 0.095, 0.047, 0.212, 0.3, 0, 0.09, 0, 0.092, 0.045, 0.05, 0.045, 0.095, 0, 0],
    "C17": [1.779, 1.89, 1.408, 1.853, 1.371, 0.815, 1.89, 1.816, 1.594, 1.964, 1.927, 1.705, 1.919, 1.89, 1.572, 1.668, 2.15, 1.964, 1.853, 1.853],
    "C18": [0.024, 0.507, 0.797, 0.58, 0.459, 0.217, 1.28, 0, 0, 0, 0, 0, 0, 0, 0.338, 0, 0, 0, 0, 0],
    "C19": [0.297, 0.95, 0.603, 0.753, 0.788, 0.574, 0.328, 0.195, 0.71, 0.176, 0.117, 0.175, 0.148, 0.166, 0.212, 0.108, 0.038, 0.135, 0.063, 0.028],
    "C20": [1.249, 1.24, 1.258, 0.992, 1.334, 1.341, 0.838, 1.377, 1.012, 1.001, 1.061, 0.856, 0.949, 1.232, 0.65, 1.112, 0.311, 0.951, 0.744, 0.539],
    "C21": [0.749, 1.075, 1.068, 1.226, 0.889, 0.986, 1.334, 1.129, 0.878, 0.663, 0.875, 1.036, 0.922, 0.972, 1.258, 0.731, None, 1.19, 0.93, 1.233],
    "C22": [3.966, 1.571, 1.034, 1.638, 2.619, 0.517, 1.379, -0.086, 1.897, 0.862, -1.039, 0.172, 0.265, 0.087, 0.517, 0.363, -0.431, 0.172, 0.086, 0],
    "C23": [0.549, 0.73, 0.402, 0.163, 0.514, 0.381, 0.303, 0.271, 0.034, -0.781, 0.058, -0.486, -0.143, 0.041, 0.226, -0.013, 0.016, -0.214, -0.209, -0.102],
    "C24": [0.46, 0.543, 0.023, 0.07, None, 0.239, -0.267, -0.221, -0.146, -0.253, -0.534, -0.046, -0.255, -0.359, -0.634, -0.103, -0.048, -0.023, -0.117, -0.07],
    "C25": [0.03, 0.529, 0.233, -0.617, 0.017, 0.032, -0.113, -0.04, 0, 0, 0, 0, 0, -0.369, 0, -0.037, 0, 0, 0, 0],
    "C26": [1.111, 1.037, 1.074, 1.074, 0.778, 0.889, 1.111, 0.963, 0.889, 0.444, 0.481, 0.926, 0.852, 0.889, 0.815, 0.63, 0.704, 0.518, 0, 0.185],
    "C27": [1.618, 1.401, 1.196, 1.002, 1.256, 1.366, 1.401, 1.023, 0.903, 0.942, 1.22, 1.26, 0.878, 1.183, 0.542, 1.285, 0.817, 0.241, 1.123, 0.089],
    "C28": [0.272, 0.761, 0.38, 0.272, 1.467, -0.761, 1.033, 1.739, 0.163, 0.87, 2.496, 0.38, 2.5, 1.304, 3.587, 2.185, 4.511, 1.685, 1.087, 2.88],
    "C29": [1.678, 1.256, 1.667, 1.853, 0.812, 1.859, 0.926, 1.132, 1.871, 1.871, 1.607, 1.871, 1.588, 1.237, 1.676, 1.135, 1.801, 1.871, 1.871, 1.871],
    "C30": [0.783, 1.833, 1.134, 0.81, 1.73, 1.135, 5.231, 1.728, 0.511, 0.349, 0.399, 0.416, 0.2, 0.672, 3.298, 0.312, 0.091, 0.272, 0.289, 0.212],
    "C31": [0.935, 0.843, 0.935, 1.037, 1.098, 0.65, 1.057, 0.874, 0.976, 0.833, 0.935, 1.077, 0.976, 0.915, 1.016, 0.828, 0.894, 0.854, 1.057, 0.874],
    "C32": [1.2, 0.3, 0.817, 1.533, 1, 1.017, 0.6, 0.467, 0.267, 0.233, 0.417, 0.1, 0.15, 1.083, 0.85, 0.267, -0.233, 0.267, 0.067, -0.167],
    "C33": [1, 0.5, 0.96, 1, 0.8, 0.56, 0.34, 0.1, 0.04, 0, 0.52, 0.3, 0.02, 0, 0.44, 0.02, 0.04, 0, 0.06, 0],
    "C34": [1.09, 0.051, 1.118, 0.815, 0.871, 0.778, 1.113, 0.266, 0.074, 0.568, 0.974, 0.55, 1.118, 1.09, 0.825, 0.708, 1.006, 0.247, 0, 0],
    "C35": [0.5, 2.5, 2.5, 0, 2.5, 1, 0.25, 0, 0, 0.75, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0],
    "C36": [0.174, -0.03, 0.21, 0.181, -0.064, 0.042, -0.085, 0.197, -0.223, -0.519, 0.803, -0.3, 0.358, 1.202, -0.001, -0.143, 0.469, -0.405, -0.282, -0.589],
    "C37": [1.036, 1.105, 0.878, 0.71, 1.174, 1.085, 1.233, 0.883, 0.69, 0.878, 1.124, 0.69, 1.213, 1.095, 0.759, 0.621, 0.957, 0.927, 0.69, 0.759],
    "C38": [0.882, 1.106, 1.296, 1.072, 1.089, 1.227, 1.21, 0.38, 0.242, 0.57, 0.156, 0.311, 1.383, 1.054, 1.089, 0.19, 0.951, 0.207, 0.121, 0.035],
    "C39": [0.809, 0.776, 0.644, 0.842, 2.295, 0.545, 0.561, 0.413, 0.594, 0.396, 0.527, 0.462, 0.495, 0.415, 0.269, 0.594, 0.314, 0.083, 0.231, 0.165],
    "C40": [0.675, 1.012, 1.166, 1.234, 0.936, 0.226, 0.901, 0.66, 0.501, 0.743, 0.518, 0.673, 0.641, 0.637, 1.054, 0.553, 0.474, 0.575, 0.824, 0.485],
    "C41": [1.19, 0.876, 1.24, 1.463, 0.694, 1.033, 1.0, 1.099, 0.876, 0.604, 1.159, 1.083, 0.868, 0.967, 1.57, 1.214, 1.364, 1.381, 0.678, 1.242],
    "C42": [1.037, 1.03, 0.847, 1.074, 0.916, 0.933, 1.151, 0.942, 0.392, 0.797, 1.037, 0.712, 1.06, 1.146, 1.098, 0.997, 0.928, 1.044, 1.046, 1.069],
    "C43": [1.003, 1.198, 0.719, 1.36, 0.414, 0.919, 1.555, 1.493, 0.35, 0.544, 1.368, 0.682, 1.559, 1.505, 1.571, 1.102, 1.274, 0.923, 1.065, 0.808],
    "C44": [0, 0, 0.042, 0.458, 0, 0, 1.208, 0, 0, 0, 1.083, 0, 1.208, 0.333, 0.613, 0.263, 0, 0, 0, 0],
    "C45": [1.14, 0.947, 1.035, 0.93, 0.895, 0.93, 0.965, 0.877, 1.07, 1.018, 0.754, 1.351, 0.996, 0.737, 0.632, 0.921, 0.93, 0.983, 1.21, 1.053],
    "C46": [1.051, 0.718, 1.359, 1.051, 1.821, 0.718, 0.179, 0.667, 0, 0.103, 0.026, 0, 0, 0.205, 0.308, 0.026, 0, 0, 0, 0],
    "C47": [1.214, 1.114, 0.8, 0.886, 0.971, 0.929, 0.757, 0.514, 0.929, 0.657, 0.543, 0.657, 0.343, 0.514, 0.514, 0.445, 0.3, 0.557, 0.471, 0.329],
    "C48": [1.409, 1.187, 0.957, 1.173, 1.168, 0.973, 0.906, 0.58, 1.135, 0.851, 0.679, 0.988, 0.445, 0.388, 0.828, 0.471, 0.145, 0.494, 0.027, 0.167],
    "C49": [1.468, 1.178, 1.035, 0.717, 0.991, 0.747, 0.963, 0.887, -0.229, -0.239, -0.365, -1.732, -0.463, 0.752, -0.378, -1.039, 0.164, -0.729, -0.757, -1.145],
    "C50": [0.841, -0.005, 1.475, 0.254, 0.415, 0.54, 0.036, 2.504, 0.977, 0.727, 0.063, 0.182, 0.151, 0.701, 0.493, 0.254, 0.01, 0.114, 0.197, 0.052],
}


def model_e(model, cid):
    return PANEL_E_ALL[cid][MODELS.index(model)]


def expired_count(model):
    """Claims whose measured effect on this model is in the expiration zone (e <= 0.2)."""
    return sum(1 for cid in PANEL_E_ALL if (v := model_e(model, cid)) is not None and v <= 0.2)


# =======================================================
# Part A: the analysis's dimension map, with evidence
# =======================================================
@dataclass(frozen=True)
class Aspect:
    name: str
    says: str            # the analysis's verdict on how this aspect evolved
    claims: tuple = ()   # study claims that test it (empty = not tested)
    flag: str = ""       # where the evidence disagrees with the analysis or needs a caveat


DIMENSIONS = {
    "A. Reasoning and problem-solving": [
        Aspect("Multi-hop / two-hop questions", "Improved, not solved", ("C1",)),
        Aspect("Reversal curse", "Still measurable", ("C13",)),
        Aspect("Premise-order sensitivity", "Failure largely expired", ("C22",),
               "expired only via the newest-model rule; the fitted curve alone says weakened"),
        Aspect("Contradictory timeline answers", "Improved, residual failures", ("C47",)),
        Aspect("Arithmetic (base 10 vs base 9)", "Major failures expired", ("C6",),
               "only counterfactual base-9 addition was tested, not arithmetic in general"),
        Aspect("Letter counting", "Major failures expired", ("C35",)),
        Aspect("Answer stability when challenged", "Major failures expired", ("C16", "C19")),
        Aspect("Overthinking easy problems", "Configuration-dependent", ("C36",)),
    ],
    "B. Knowledge retrieval and context": [
        Aspect("Long-context retrieval / position", "Lost-in-the-middle expired", ("C7",)),
        Aspect("Factual recall in general", "Not established by the study"),
        Aspect("Knowledge integration", "Residual multi-hop errors", ("C1",)),
        Aspect("Temporal consistency", "Improved, not eliminated", ("C47",)),
        Aspect("Directional relations", "Reversal curse, weakened", ("C13",)),
        Aspect("Niche-domain knowledge", "Residual hallucination", ("C20", "C30")),
        Aspect("Multilingual behaviour", "Some disparities gone", ("C3", "C50")),
    ],
    "C. Accuracy, factuality, hallucination": [
        Aspect("Citation fabrication", "Largely expired", ("C11",),
               "panel-wide verdict; Qwen3.7-Flash still fabricates 69%"),
        Aspect("Bibliographic accuracy", "Separate issue", (),
               "explicitly not measured: only nonexistent works counted"),
        Aspect("Legal hallucination", "Weakened, still measurable", ("C20",)),
        Aspect("Package hallucination", "Weakened", ("C30",)),
        Aspect("Answering from answerless documents", "Related concerns remain", ("C10",)),
        Aspect("Appropriate abstention", "Still problematic", ("C41",)),
    ],
    "D. Instruction following and controllability": [
        Aspect("Multi-turn instruction following", "Weakened failure", ("C39",)),
        Aspect("Prompt-injection resistance", "Improved, residual", ("C2", "C25", "C34")),
        Aspect("Jailbreak resistance", "Improved, residual", ("C26",)),
        Aspect("User-pressure resistance", "Major failure expired", ("C16", "C19", "C38"),
               "stating falsehoods under pressure (C38) has only weakened"),
        Aspect("Refusal calibration", "Mixed", ("C8", "C32"),
               "both over-refusal claims tested are expired"),
        Aspect("Task completion", "Not measured"),
    ],
    "E. Security and safety": [
        Aspect("Jailbreak susceptibility", "Weakened", ("C26",)),
        Aspect("Prompt-injection susceptibility", "Weakened", ("C2", "C25", "C34")),
        Aspect("Blackmail in agentic scenarios", "Model-dependent, uncertain", ("C44",),
               "verdict expired, but support is weak; several recent models still blackmail"),
        Aspect("Privacy leakage (meeting summaries)", "Historical failure expired", ("C46",)),
        Aspect("Harmful / illogical compliance", "Tied to refusal calibration", ("C33",)),
        Aspect("Deceptive behaviour", "Some tendencies persist", ("C18", "C38"),
               "insider trading (C18) expired; lying under pressure (C38) weakened"),
    ],
    "F. Social behaviour, bias, interaction": [
        Aspect("Sycophancy (all sycophancy claims)", "Persists or strengthened",
               ("C16", "C19", "C32", "C33", "C40", "C42", "C43", "C17"),
               "only half persist or grow: flattery persists, caving on facts expired"),
        Aspect("Praise follows user preference", "Strengthened", ("C17",)),
        Aspect("Affirming the user's actions", "Persists", ("C42", "C43")),
        Aspect("Political lean", "Left lean persists", ("C21",)),
        Aspect("Gender stereotypes", "Weakened", ("C9",)),
        Aspect("Implicit association bias", "Weakened", ("C48",)),
        Aspect("Dialect bias (AAE)", "Expired, trend reverses", ("C24",)),
        Aspect("Language-dependent answers", "Mixed", ("C3", "C50")),
        Aspect("Persona stereotypes", "Some effects expired", ("C5",)),
    ],
    "G. Evaluation, judgment, self-assessment": [
        Aspect("Self-preference", "Strengthened", ("C28",),
               "from a small 4.6 pp reference; huge spread between models"),
        Aspect("Length preference", "Weakened, model-dependent", ("C27",)),
        Aspect("Position / order bias", "Historical effect remains", ("C4", "C12"),
               "MCQ letter bias (C12) expired; order-swap flips (C4) weakened"),
        Aspect("Peer-review discrimination", "Some failures persist", ("C31", "C34")),
        Aspect("Swayed by fabricated references", "Needs further evaluation", ("C23",),
               "the study tested it: expired"),
        Aspect("Rating spun abstracts higher", "Persists", ("C37",)),
        Aspect("Preferring AI-written text", "(not in the analysis)", ("C49",),
               "added: expired, the fitted trend reverses"),
        Aspect("Judge reliability vs humans", "Mixed"),
    ],
    "H. Diversity, randomness, generation": [
        Aspect("Output / sampling diversity", "Near-identical outputs persist", ("C45",)),
        Aspect("Random-number generation", "Non-uniformity strengthened", ("C29",)),
        Aspect("Creative variation", "Not measured"),
        Aspect("Preference convergence", "Persistent", ("C45",)),
        Aspect("Similarity between models", "Not measured"),
    ],
}


def evidence(aspect):
    """[(cid, verdict, e_now)] for the claims that test this aspect."""
    return [(cid, CLAIM[cid].verdict, CLAIM[cid].e_now) for cid in aspect.claims]


def map_summary():
    """How much of the analysis's map the study actually tests."""
    rows = [a for asp in DIMENSIONS.values() for a in asp]
    return dict(rows=len(rows), tested=sum(bool(a.claims) for a in rows),
                flagged=sum(bool(a.flag) for a in rows))


# Our split of the 50 claims into "has one right answer" vs "tendency / preference".
# The paper draws this line informally ("what models cannot do" vs "what they prefer").
TENDENCY = {"C3", "C4", "C5", "C9", "C10", "C12", "C17", "C21", "C23", "C24", "C27", "C28",
            "C29", "C31", "C37", "C40", "C41", "C42", "C43", "C45", "C48", "C49", "C50"}


def class_counts():
    out = {"capability failure": Counter(), "behavioural tendency": Counter()}
    for c in CLAIMS:
        out["behavioural tendency" if c.cid in TENDENCY else "capability failure"][c.verdict] += 1
    return out


# =======================================================
# Part A: the performance vector P(M_t)
# =======================================================
AREAS = list(AREA_SHORT)          # the study's nine research areas
AREA_CLAIMS = {a: [c.cid for c in CLAIMS if c.area == a] for a in AREAS}


def performance_vector(model):
    """P_area(M) = share of the area's published failure/bias claims that no longer
    hold on model M (measured e <= 0.2). 1.0 = every documented problem is gone."""
    vec = {}
    for area, cids in AREA_CLAIMS.items():
        vals = [v for cid in cids if (v := model_e(model, cid)) is not None]
        vec[area] = sum(v <= 0.2 for v in vals) / len(vals) if vals else float("nan")
    return vec


def delta_p(m1, m2):
    p1, p2 = performance_vector(m1), performance_vector(m2)
    return {a: p2[a] - p1[a] for a in AREAS}


def improvement_rates():
    """Slope of P_area vs release date across the 20 models (percentage points / year)."""
    t = np.array([(RELEASE[m] - RELEASE[MODELS[0]]).days / 365.25 for m in MODELS])
    rates = {}
    for a in AREAS:
        y = np.array([performance_vector(m)[a] for m in MODELS]) * 100
        rates[a] = float(np.polyfit(t, y, 1)[0])
    return rates


# =======================================================
# Part A: a longitudinal capability database (the analysis's proposal)
# =======================================================
SCHEMA = """
-- one row per claim/test, frozen forever
CREATE TABLE benchmark (
  id TEXT PRIMARY KEY, area TEXT, title TEXT,
  source TEXT, published TEXT,
  prompt_sha TEXT, scorer TEXT);
CREATE TABLE model (
  name TEXT PRIMARY KEY, release TEXT);
-- one row per (model, benchmark, run)
CREATE TABLE result (
  model TEXT, benchmark TEXT, e REAL,
  eval_date TEXT, config TEXT);
"""


def build_db():
    db = sqlite3.connect(":memory:")
    db.executescript(SCHEMA)
    for c in CLAIMS:
        sha = hashlib.sha256(c.title.encode()).hexdigest()[:12]   # stand-in for prompt files
        db.execute("INSERT INTO benchmark VALUES (?,?,?,?,?,?,?)",
                   (c.cid, AREA_SHORT[c.area], c.title, c.cite, c.published, sha,
                    "original paper scorer"))
    db.executemany("INSERT INTO model VALUES (?,?)", PANEL)
    db.executemany("INSERT INTO result VALUES (?,?,?,?,?)",
                   [(m, cid, v, "2026-10", "provider default")
                    for cid, vals in PANEL_E_ALL.items() for m, v in zip(MODELS, vals)
                    if v is not None])
    return db


TREND_SQL = """
SELECT b.area,
  ROUND(AVG(CASE WHEN m.release < '2025-07'
            THEN r.e <= 0.2 END), 2) AS early,
  ROUND(AVG(CASE WHEN m.release >= '2025-07'
            THEN r.e <= 0.2 END), 2) AS late
FROM result r
  JOIN benchmark b ON b.id = r.benchmark
  JOIN model m ON m.name = r.model
GROUP BY b.area ORDER BY late - early DESC;
"""


def db_trend():
    return build_db().execute(TREND_SQL).fetchall()


# =======================================================
# Terminal views
# =======================================================
def print_map():
    for dim, rows in DIMENSIONS.items():
        print(f"\n{dim}")
        for a in rows:
            ev = ", ".join(f"{cid} {v} ({e:.2f})" for cid, v, e in evidence(a)) or "not tested"
            print(f"  {a.name:<38} analysis: {a.says:<30} study: {ev}")
            if a.flag:
                print(f"  {'':<38} ! {a.flag}")
    s = map_summary()
    print(f"\n{s['rows']} aspects, {s['tested']} tested by the study, {s['flagged']} flagged")


def print_demo():
    print("Claims in the expiration zone per model (check against the paper: 5 ... 30):")
    for m in MODELS:
        print(f"  {m:<24} {RELEASE[m]}  {expired_count(m):>2}")
    print("\nDelta P, GPT-4o mini -> GPT-6.1 Sol (share of documented problems gone):")
    for a, d in sorted(delta_p("GPT-4o mini", "GPT-6.1 Sol").items(), key=lambda kv: -kv[1]):
        print(f"  {AREA_SHORT[a]:<28} {d:+.2f}")
    print("\nImprovement rate across the panel (pp / year):")
    for a, r in sorted(improvement_rates().items(), key=lambda kv: -kv[1]):
        print(f"  {AREA_SHORT[a]:<28} {r:+5.1f}")
    print("\nClass split:", {k: dict(v) for k, v in class_counts().items()})
    print("\nLongitudinal DB, share expired before / after mid-2025:")
    for row in db_trend():
        print("  ", row)


# =======================================================
# Part B: slides (helpers and styling from 9_15)
# =======================================================
plt = base.plt
slide, blank, style, round_box, card = base.slide, base.blank, base.style, base.round_box, base.card
text_block, row_list, code_panel, footnote = (base.text_block, base.row_list, base.code_panel,
                                              base.footnote)
FS_TITLE, FS_HEAD, FS_BODY, FS_NOTE = base.FS_TITLE, base.FS_HEAD, base.FS_BODY, base.FS_NOTE
V_COLOR, V_FACE, C_DARK, C_MUTE, C_ACCENT = (base.V_COLOR, base.V_FACE, base.C_DARK,
                                             base.C_MUTE, base.C_ACCENT)
C_FLAG = "#b03a2e"


def source_of(obj):
    return textwrap.dedent(inspect.getsource(obj))


# -------------------------------------------------------
# Preface and central idea
# -------------------------------------------------------
def build_preface():
    fig, gs = slide("")
    ax = fig.add_subplot(gs[0, 0])
    blank(ax)
    ax.text(6, 9.3, "Analysis of", ha="center", fontsize=FS_HEAD + 2, color="0.45",
            fontweight="bold")
    round_box(ax, 0.6, 5.9, 10.8, 2.8, "#d6eaf8", C_DARK, lw=2.2)
    ax.text(6, 7.7, "LLM performance over time", ha="center", va="center",
            fontsize=base.FS_FIGURE_TITLE + 10, fontweight="bold", color=C_DARK)
    ax.text(6, 6.55, "a multidimensional, time-dependent function", ha="center",
            va="center", fontsize=FS_HEAD + 4, color="0.25")
    ax.text(6, 4.85, "Based on Queen & Zou, \"The shelf life of LLM findings\", Stanford, "
            "Nov 2, 2026", ha="center", fontsize=FS_HEAD + 1, fontweight="bold")
    ax.text(6, 3.8, "Follows a written analysis of the study (GPT-assisted notes),\n"
            "checked claim by claim against the study's own data", ha="center",
            fontsize=FS_HEAD, color="0.3", linespacing=1.5)
    ax.text(6, 1.9, "Companion deck: 9_15_LLM_findings_Claude.py (the study itself)", ha="center",
            fontsize=FS_BODY, color="0.45")
    return fig


def build_step_1():
    fig, gs = slide("Step 1: The central idea", ncols=2, width_ratios=[1.15, 1], wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    blank(ax)
    round_box(ax, 0.2, 6.6, 11.6, 3.2, "#d6eaf8", C_DARK, lw=2)
    text_block(ax, 0.6, 9.4, "An empirical finding about an LLM\nis not a permanent property "
               "of LLMs.\nIt may describe one generation\nof models.", fs=FS_HEAD + 1,
               fontweight="bold", color=C_DARK)
    text_block(ax, 0.4, 5.9,
               "50 published findings  ·  20 models  ·  8 developers\n"
               "released July 2024 - September 2026\n\n"
               "This is not a leaderboard of LLM quality.\n"
               "It measures whether specific behavioural\n"
               "findings still hold.", fs=FS_BODY)
    ax = fig.add_subplot(gs[0, 1])
    blank(ax)
    vc = Counter(c.verdict for c in CLAIMS)
    for i, v in enumerate(VERDICTS):
        y = 9.3 - i * 2.3
        round_box(ax, 0.4, y - 1.9, 11.2, 1.9, V_FACE[v], V_COLOR[v], lw=2)
        ax.text(1.0, y - 0.95, f"{vc[v] / len(CLAIMS):.0%}", va="center",
                fontsize=FS_TITLE + 4, fontweight="bold", color=V_COLOR[v])
        ax.text(4.6, y - 0.95, f"{v}\n{vc[v]} of {len(CLAIMS)} claims", va="center",
                fontsize=FS_HEAD, linespacing=1.4)
    return fig


# -------------------------------------------------------
# Dimensions A-H
# -------------------------------------------------------
DIM_COLORS = ["#1a5276", "#117a65", "#7d3c98", "#b9770e", "#922b21", "#1f618d", "#6e2c00",
              "#196f3d"]


def build_step_2():
    fig, gs = slide("Step 2: Eight dimensions of LLM performance")
    ax = fig.add_subplot(gs[0, 0])
    blank(ax, xlim=(0, 24))
    most = max(sum(len(a.claims) for a in rows) for rows in DIMENSIONS.values())
    for i, (dim, rows) in enumerate(DIMENSIONS.items()):
        col, row = i % 4, i // 4
        x, y = 0.2 + col * 6, 9.6 - row * 4.9
        verdicts = Counter(v for a in rows for _, v, _ in evidence(a))
        tested = sum(bool(a.claims) for a in rows)
        round_box(ax, x, y - 4.4, 5.6, 4.4, "white", DIM_COLORS[i], lw=2)
        ax.text(x + 0.25, y - 0.35, dim.split(". ")[0], fontsize=FS_HEAD + 6,
                fontweight="bold", color=DIM_COLORS[i], va="top")
        ax.text(x + 1.65, y - 0.45, textwrap.fill(dim.split(". ")[1], 17), fontsize=FS_NOTE + 1,
                fontweight="bold", va="top", linespacing=1.2)
        ax.text(x + 0.3, y - 2.2, f"{len(rows)} aspects, {tested} tested", fontsize=FS_NOTE,
                color="0.3", va="top")
        xx = x + 0.3
        for v in VERDICTS:
            if verdicts[v]:
                w = 5.0 * verdicts[v] / most
                ax.add_patch(plt.Rectangle((xx, y - 3.9), w, 0.8, color=V_COLOR[v]))
                if w > 0.5:
                    ax.text(xx + w / 2, y - 3.5, str(verdicts[v]), ha="center", va="center",
                            color="white", fontsize=FS_NOTE, fontweight="bold")
                xx += w
    footnote(fig, "Bars: verdicts of the claims cited as evidence (grey expired, orange "
                  "weakened, blue persists, red strengthened).")
    return fig


def make_dimension_slide(k, dim):
    rows = DIMENSIONS[dim]

    def build():
        fig, gs = slide(f"Step {3 + k}: {dim}", top=0.86, bottom=0.12)
        ax = fig.add_subplot(gs[0, 0])
        blank(ax, xlim=(0, 24), ylim=(0, 10))
        step = min(1.1, 9.0 / len(rows))
        heads = [(0.2, "Aspect"), (8.2, "Analysis says"), (14.6, "Study evidence (fitted e)")]
        for x, h in heads:
            ax.text(x, 9.75, h, fontsize=FS_BODY, fontweight="bold", color=DIM_COLORS[k],
                    va="center")
        ax.plot([0.1, 23.9], [9.4, 9.4], color=DIM_COLORS[k], lw=1.5)
        for i, a in enumerate(rows):
            y = 9.4 - (i + 0.5) * step
            if a.flag:
                ax.add_patch(plt.Rectangle((0.1, y - step / 2 + 0.04), 23.8, step - 0.08,
                                           color="#fdedec", lw=0))
            ax.text(0.2, y + (0.15 if a.flag else 0), a.name, fontsize=FS_BODY, va="center",
                    fontweight="bold")
            ax.text(8.2, y + (0.15 if a.flag else 0), a.says, fontsize=FS_BODY, va="center")
            ev = evidence(a)
            if not ev:
                ax.text(14.6, y + (0.15 if a.flag else 0), "not tested in the study",
                        fontsize=FS_NOTE, va="center", color="0.5", style="italic")
            for j, (cid, v, e) in enumerate(ev[:4]):
                x = 14.6 + j * 2.3
                ax.text(x, y + (0.15 if a.flag else 0), f"{cid} {e:.2f}", fontsize=FS_NOTE,
                        va="center", color=V_COLOR[v], fontweight="bold")
            if len(ev) > 4:
                n = Counter(v for _, v, _ in ev)
                ax.text(14.6, y - 0.3, "  ".join(f"{n[v]} {v}" for v in VERDICTS if n[v]),
                        fontsize=FS_NOTE - 1, va="center", color="0.3")
            if a.flag:
                ax.text(0.5, y - 0.3, "! " + a.flag, fontsize=FS_NOTE - 1, va="center",
                        color=C_FLAG, style="italic")
        fig.text(0.5, 0.05, "Colour = study verdict: grey expired · orange weakened · "
                 "blue persists · red strengthened.   Pink rows: evidence adds a caveat.",
                 ha="center", fontsize=FS_NOTE, color="0.35", style="italic")
        return fig

    build.__name__ = f"build_dim_{dim[0]}"
    return build


DIM_BUILDERS = [make_dimension_slide(k, d) for k, d in enumerate(DIMENSIONS)]


# -------------------------------------------------------
# Fact check, causation, critique
# -------------------------------------------------------
def build_step_11():
    fig, gs = slide("Step 11: Where the analysis needs correcting")
    ax = fig.add_subplot(gs[0, 0])
    blank(ax)
    syc = Counter(CLAIM[c].verdict for c in DIMENSIONS["F. Social behaviour, bias, interaction"][0].claims)
    items = [
        ("'Sycophancy persists or strengthened' is half true",
         f"8 sycophancy claims: {syc['expired']} expired, {syc['weakened']} weakened, "
         f"{syc['persists']} persist, {syc['strengthened']} strengthened.  Caving on facts "
         "faded; flattery did not"),
        ("'Fabricated evidence: requires further evaluation'",
         "The study tested it (C23, fake references sway a judge): expired, e = -0.10"),
        ("'Position bias: historical effect remains'",
         "MCQ letter bias (C12) expired; answer-order flips (C4) weakened to e = 0.53"),
        ("'Refusal calibration: mixed'",
         "Both over-refusal claims tested (C8, C32) are expired"),
        ("'Arithmetic: major failures expired'",
         "Only base-9 vs base-10 addition (C6) was tested; nothing about arithmetic in general"),
        ("'Outlier exclusion may conceal weaknesses'",
         "Fair concern, but the paper reports that including outliers changes no verdict"),
    ]
    row_list(ax, items, top=9.9, step=1.6, face="#fdedec", edge=C_FLAG)
    return fig


def build_step_12():
    fig, gs = slide("Step 12: Correlation with release date is not a causal effect", ncols=2,
                    width_ratios=[1, 1], wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    blank(ax)
    round_box(ax, 0.3, 7.0, 11.4, 2.6, "#fdebd0", "#e67e22", lw=2)
    ax.text(6, 8.3, "trend vs release date  ≠  effect of\n"
            "a specific development intervention", ha="center", va="center",
            fontsize=FS_HEAD + 2, fontweight="bold", linespacing=1.4)
    text_block(ax, 0.4, 6.3,
               "The study shows measured behaviours change\n"
               "across model generations.\n\n"
               "It does not show which change caused it:\n"
               "  architecture, data, scale, post-training,\n"
               "  reasoning budget, or a targeted patch.", fs=FS_BODY)
    ax = fig.add_subplot(gs[0, 1])
    blank(ax)
    text_block(ax, 0.3, 9.6, "Confounders inside the panel", fs=FS_HEAD, fontweight="bold")
    text_block(ax, 0.3, 8.7,
               "Developer      8 labs, different recipes\n"
               "Size           small and large models mixed\n"
               "Openness       open- and closed-weight\n"
               "Config         provider defaults; reasoning\n"
               "               effort not equalised\n"
               "Exposure       famous failures may be\n"
               "               patched specifically",
               fs=FS_BODY, family="monospace")
    footnote(fig, "The authors fit open vs closed and small vs large separately and note splits "
                  "on each claim card - a partial control, not a causal design.")
    return fig


LIMITS = [
    ("Only 50 claims", "limited coverage", "authors: intervals are wide because n = 50"),
    ("Only 20 models", "thin panel", "four 2024 models anchor the early cohorts"),
    ("Default API configs", "capability vs setting", "acknowledged; GPT-5.2 used no reasoning on C36"),
    ("Release date as time", "no mechanism", "mechanisms offered only as hypotheses"),
    ("One minimal design / claim", "test-specific gains", "acknowledged as a limit of the panel"),
    ("Substitute models", "weaker reproduction", "25 of 48 exact; 43 of 48 reproduce"),
    ("Curve fitting", "shapes are approximations", "4 shapes by AICc, 500 bootstraps; 3 fits flagged"),
    ("Outlier exclusion", "may hide weak models", "<= 20% flagged; including them changes no label"),
    ("Size vs release date", "confounded", "open/closed and small/large fitted separately"),
]


def build_step_13():
    fig, gs = slide("Step 13: Critical evaluation - and what the paper already reports",
                    top=0.86, bottom=0.12)
    ax = fig.add_subplot(gs[0, 0])
    blank(ax, xlim=(0, 24))
    for x, h in [(0.2, "Limitation"), (7.6, "Why it matters"), (13.4, "What the paper reports")]:
        ax.text(x, 9.75, h, fontsize=FS_BODY, fontweight="bold", color=C_DARK, va="center")
    ax.plot([0.1, 23.9], [9.4, 9.4], color=C_DARK, lw=1.5)
    for i, (lim, why, rep) in enumerate(LIMITS):
        y = 9.4 - (i + 0.5) * 1.0
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((0.1, y - 0.5), 23.8, 1.0, color=C_MUTE, lw=0))
        ax.text(0.2, y, lim, fontsize=FS_BODY, fontweight="bold", va="center")
        ax.text(7.6, y, why, fontsize=FS_BODY, va="center")
        ax.text(13.4, y, rep, fontsize=FS_BODY, va="center", color="0.25")
    footnote(fig, "The limitations are real; most are disclosed.  The confidence intervals "
                  "do not remove them.", y=0.05)
    return fig


# -------------------------------------------------------
# Two classes, development trajectory
# -------------------------------------------------------
def build_step_14():
    fig, gs = slide("Step 14: Capability failures vs behavioural tendencies (live)", ncols=2,
                    width_ratios=[1.3, 1], wspace=0.12, left=0.2)
    cc = class_counts()
    ax = fig.add_subplot(gs[0, 0])
    for i, (cls, cnt) in enumerate(cc.items()):
        n = sum(cnt.values())
        left = 0
        for v in VERDICTS:
            w = 100 * cnt[v] / n
            ax.barh(i, w, left=left, color=V_COLOR[v], height=0.55)
            if cnt[v]:
                ax.text(left + w / 2, i, f"{cnt[v]}", ha="center", va="center", color="white",
                        fontsize=FS_BODY, fontweight="bold")
            left += w
    ax.set_yticks([0, 1], [f"{k}\n({sum(v.values())} claims)" for k, v in cc.items()],
                  fontsize=FS_BODY)
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    style(ax, xlabel="% of claims in the class")
    ax.spines[["top", "right"]].set_visible(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=V_COLOR[v], label=v) for v in VERDICTS], ncol=4,
              loc="upper center", bbox_to_anchor=(0.5, 1.1), fontsize=FS_NOTE, frameon=False)
    ax = fig.add_subplot(gs[0, 1])
    blank(ax)
    text_block(ax, 0.3, 9.6, "Capability failure", fs=FS_HEAD, fontweight="bold",
               color=V_COLOR["expired"])
    text_block(ax, 0.3, 8.8, "one right answer -> can be\nmeasured, rewarded, trained",
               fs=FS_BODY)
    text_block(ax, 0.3, 6.8, "Behavioural tendency", fs=FS_HEAD, fontweight="bold",
               color=V_COLOR["strengthened"])
    text_block(ax, 0.3, 6.0, "no universal correct behaviour:\nsycophancy, self-preference,\n"
               "sameness, political lean, bias", fs=FS_BODY)
    text_block(ax, 0.3, 3.4, "The mechanism (post-training\nremoves measurable failures) is\n"
               "the authors' hypothesis, not\nsomething the study isolates.", fs=FS_NOTE,
               color="0.35", style="italic")
    footnote(fig, "Class labels are ours (TENDENCY set in Part A); the paper makes this split "
                  "informally.")
    return fig


def build_step_15():
    fig, gs = slide("Step 15: A possible development trajectory (interpretation)")
    ax = fig.add_subplot(gs[0, 0])
    blank(ax, xlim=(0, 24))
    stages = [
        ("1", "Foundation model", "language modelling, representations, knowledge"),
        ("2", "Supervised post-training", "instruction following; train against known failures"),
        ("3", "Preference optimisation", "helpfulness, interaction quality"),
        ("4", "Reasoning-oriented", "multi-step problem solving"),
        ("5", "Agentic", "tools, retrieval, code execution, workflows"),
        ("6", "Evaluation-driven refinement", "automated tests and agents find and fix weaknesses"),
    ]
    for i, (n, head, sub) in enumerate(stages):
        x = 0.3 + i * 3.95
        round_box(ax, x, 3.6, 3.6, 5.6, "#eaf2f8", C_DARK, lw=1.8)
        ax.text(x + 1.8, 8.6, n, ha="center", fontsize=FS_TITLE, fontweight="bold",
                color=C_DARK, va="center")
        ax.text(x + 1.8, 7.4, textwrap.fill(head, 14), ha="center", va="top",
                fontsize=FS_BODY, fontweight="bold", linespacing=1.2)
        ax.text(x + 1.8, 5.6, textwrap.fill(sub, 18), ha="center", va="top", fontsize=FS_NOTE,
                color="0.25", linespacing=1.25)
        if i < 5:
            ax.annotate("", xy=(x + 3.95, 6.4), xytext=(x + 3.6, 6.4),
                        arrowprops=dict(arrowstyle="->", lw=2, color=C_DARK))
    text_block(ax, 0.4, 2.7,
               "Links to the study:\n"
               "  stages 2-3 match 'post-training removes measurable failures' - and may also "
               "reinforce agreeable tendencies\n"
               "  stage 4 fits the expired reasoning claims\n"
               "  stage 6 is how the study itself was done (coding agents reproduced 48 claims)",
               fs=FS_BODY)
    footnote(fig, "This chronology is the analysis's interpretation of LLM development, not "
                  "something the paper tested.", y=0.07)
    return fig


# -------------------------------------------------------
# Performance vector framework
# -------------------------------------------------------
def build_step_16():
    fig, gs = slide("Step 16: Performance as a vector, not one score (live)", ncols=2,
                    width_ratios=[1, 1.1], wspace=0.55, left=0.04)
    ax = fig.add_subplot(gs[0, 0])
    blank(ax)
    text_block(ax, 0.3, 9.6, "P(M_t) = [P_reasoning,\n"
               "          P_context, ...,\n"
               "          P_diversity]\n\n"
               "ΔP = P(M_t2) - P(M_t1)", fs=FS_BODY + 1, family="monospace")
    text_block(ax, 0.3, 5.8,
               "Here, with the study's data:\n"
               "P_area(M) = share of that area's\n"
               "published problems that no longer\n"
               "hold on model M (e ≤ 0.2).\n\n"
               "Right: ΔP from GPT-4o mini\n"
               "(Jul 2024) to GPT-6.1 Sol\n(Sep 2026).", fs=FS_BODY)
    ax = fig.add_subplot(gs[0, 1])
    d = delta_p("GPT-4o mini", "GPT-6.1 Sol")
    order = sorted(AREAS, key=lambda a: d[a])
    vals = [100 * d[a] for a in order]
    ax.barh(range(len(order)), vals, color=[C_DARK if v > 0 else C_ACCENT for v in vals],
            height=0.6)
    for i, v in enumerate(vals):
        ax.text(v + (1.5 if v >= 0 else -1.5), i, f"{v:+.0f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=FS_BODY, fontweight="bold")
    ax.set_yticks(range(len(order)), [AREA_SHORT[a] for a in order], fontsize=FS_BODY)
    ax.axvline(0, color="0.3", lw=1)
    ax.set_xlim(-25, 105)
    style(ax, "ΔP: share of documented problems gone (pp)")
    ax.spines[["top", "right"]].set_visible(False)
    footnote(fig, f"Expired claims overall: {expired_count('GPT-4o mini')} of 50 on GPT-4o mini "
                  f"-> {expired_count('GPT-6.1 Sol')} of 50 on GPT-6.1 Sol (matches the paper).")
    return fig


def build_step_16_1():
    fig, gs = slide("Step 16.1: The vector in code (Part A)", top=0.87, bottom=0.13)
    code_panel(fig.add_subplot(gs[0, 0]),
               source_of(performance_vector) + "\n" + source_of(delta_p) + "\n"
               + source_of(improvement_rates), title="performance_vector(), delta_p(), "
               "improvement_rates()")
    return fig


def build_step_17():
    fig, gs = slide("Step 17: Q1 - Do the dimensions improve at the same rate? (live)",
                    left=0.25, right=0.95)
    rates = improvement_rates()
    order = sorted(AREAS, key=lambda a: rates[a])
    ax = fig.add_subplot(gs[0, 0])
    vals = [rates[a] for a in order]
    ax.barh(range(len(order)), vals, color=[C_DARK if v > 0 else C_ACCENT for v in vals],
            height=0.6)
    for i, v in enumerate(vals):
        ax.text(v + (0.6 if v >= 0 else -0.6), i, f"{v:+.1f} pp/yr", va="center",
                ha="left" if v >= 0 else "right", fontsize=FS_BODY)
    ax.set_yticks(range(len(order)), [AREA_SHORT[a] for a in order], fontsize=FS_BODY)
    ax.axvline(0, color="0.3", lw=1)
    lo, hi = min(vals + [0]), max(vals)
    ax.set_xlim(lo - 8, hi + 10)
    style(ax, xlabel="linear slope of P_area vs model release date, 20 models (pp per year)")
    ax.spines[["top", "right"]].set_visible(False)
    footnote(fig, "No.  Safety, bias, sycophancy, judges ~20 pp/yr; factuality ~3; diversity "
                  "0.  (20 models from 8 labs: indicative only.)")
    return fig


def build_step_18():
    fig, gs = slide("Step 18: Model identity matters - P_area for every model", left=0.2,
                    right=0.97, top=0.88, bottom=0.33)
    ax = fig.add_subplot(gs[0, 0])
    M = np.array([[performance_vector(m)[a] for m in MODELS] for a in AREAS])
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=FS_NOTE - 3,
                    color="white" if M[i, j] > 0.6 else "0.2")
    ax.set_yticks(range(len(AREAS)), [AREA_SHORT[a] for a in AREAS], fontsize=FS_NOTE)
    short = {"Llama 3.1 70B Instruct": "Llama 3.1 70B", "Qwen2.5-72B-Instruct": "Qwen2.5-72B",
             "Mistral Small 3 (2501)": "Mistral Small 3", "DeepSeek V4 Pro 0813": "DeepSeek V4 Pro"}
    ax.set_xticks(range(len(MODELS)), [f"{short.get(m, m)} ({expired_count(m)})" for m in MODELS],
                  rotation=55, ha="right", fontsize=FS_NOTE - 2)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    footnote(fig, "Models sorted by release date; (n) = claims expired on that model.  "
                  "Same quarter, 11 (DeepSeek V4 Pro) to 30 (GPT-6.1 Sol).", y=0.03)
    return fig


def build_step_19():
    fig, gs = slide("Step 19: Q2 trade-offs and Q3 generalising vs patching", ncols=2,
                    wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    blank(ax)
    text_block(ax, 0.2, 9.6, "Q2  Possible trade-offs", fs=FS_HEAD, fontweight="bold",
               color=C_DARK)
    row_list(ax, [("More reasoning", "vs inference cost and latency"),
                  ("Stronger safety", "vs excessive refusals"),
                  ("Instruction compliance", "vs prompt-injection vulnerability"),
                  ("Answer confidence", "vs calibration and abstention")],
             top=8.7, step=1.75, x=0.2, w=11.4)
    ax = fig.add_subplot(gs[0, 1])
    blank(ax)
    text_block(ax, 0.2, 9.6, "Q3  General skill or targeted patch?", fs=FS_HEAD,
               fontweight="bold", color=C_DARK)
    text_block(ax, 0.2, 8.6,
               "A model can pass one famous test without\n"
               "gaining the underlying skill.\n\n"
               "Hints in the study:\n"
               "  · each claim uses one minimal design\n"
               "  · high-profile failures (strawberry,\n"
               "    'Are you sure?') may be patched\n"
               "  · two-hop: still ~10% wrong on Claude\n"
               "    Sonnet 5.5 although both sub-answers\n"
               "    are right\n\n"
               "Test: re-run with fresh variants of the\n"
               "same task, not only the published prompt.", fs=FS_BODY)
    footnote(fig, "The study does not test these trade-offs; they are research questions "
                  "raised by the analysis.")
    return fig


EXTRA_DIMS = [
    ("Language", "fluency, multilingual: human eval, per-language accuracy"),
    ("Knowledge", "breadth, freshness: domain coverage, time-dependent facts"),
    ("Reasoning", "math, logic, causal, abstract: accuracy, contradiction rate"),
    ("Coding", "generation, debugging, repo tasks: pass@k, fix rate, runtime"),
    ("Context", "utilisation, long-context reasoning: accuracy by position / length"),
    ("Generation", "diversity, creativity: entropy, distinct-n, self-BLEU"),
    ("Reliability", "factuality, calibration, abstention: Brier, ECE, selective risk"),
    ("Safety", "robustness, privacy: attack success, leakage rate"),
    ("Agents", "tools, planning, recovery: task success, interventions per task"),
    ("Efficiency", "speed, latency, cost: tokens/s, TTFT, cost per solved task"),
    ("Multimodal", "image, audio, video: VQA accuracy, WER, temporal reasoning"),
]


def build_step_20():
    fig, gs = slide("Step 20: Dimensions a full evaluation would add (not in the study)")
    ax = fig.add_subplot(gs[0, 0])
    blank(ax, xlim=(0, 24))
    for i, (cat, txt) in enumerate(EXTRA_DIMS):
        y = 9.6 - i * 0.86
        round_box(ax, 0.2, y - 0.36, 3.4, 0.72, C_DARK, C_DARK, lw=0)
        ax.text(1.9, y, cat, ha="center", va="center", color="white", fontsize=FS_BODY,
                fontweight="bold")
        ax.text(4.0, y, txt, va="center", fontsize=FS_BODY)
    footnote(fig, "Proposed by the analysis; none of these are results of the Stanford study.",
             y=0.07)
    return fig


# -------------------------------------------------------
# Longitudinal database, conclusions
# -------------------------------------------------------
def build_step_21():
    fig, gs = slide("Step 21: A longitudinal capability database (live, SQLite)", ncols=2,
                    width_ratios=[1.05, 1], wspace=0.06, top=0.87, bottom=0.13)
    code_panel(fig.add_subplot(gs[0, 0]), SCHEMA.strip() + "\n" + TREND_SQL.strip(),
               title="schema + one longitudinal query")
    ax = fig.add_subplot(gs[0, 1])
    blank(ax, xlim=(0, 12), ylim=(0, 10))
    rows = db_trend()
    ax.text(0.2, 9.7, "Share of claims expired, per area", fontsize=FS_BODY,
            fontweight="bold", va="top")
    ax.text(7.2, 9.0, "≤ mid-2025", fontsize=FS_NOTE, ha="center", color="0.35")
    ax.text(10.2, 9.0, "after", fontsize=FS_NOTE, ha="center", color="0.35")
    for i, (area, before, after) in enumerate(rows):
        y = 8.0 - i * 0.8
        ax.text(0.2, y, area, fontsize=FS_NOTE, va="center")
        ax.text(7.2, y, f"{before:.2f}", fontsize=FS_NOTE, va="center", ha="center")
        ax.text(10.2, y, f"{after:.2f}", fontsize=FS_NOTE, va="center", ha="center",
                fontweight="bold", color=C_DARK)
    ax.text(0.2, 0.4, "Each row keeps prompts (hash), scorer, model version,\n"
            "release date, eval date and config - so a claim can be re-run.",
            fontsize=FS_NOTE, color="0.3", va="bottom")
    return fig


def build_step_22():
    fig, gs = slide("Step 22: Conclusions")
    ax = fig.add_subplot(gs[0, 0])
    blank(ax, xlim=(0, 24))
    items = [
        ("1", "Capabilities move fast: published results are not permanent descriptions."),
        ("2", "Failures with a right answer are easiest to remove."),
        ("3", "Tendencies (flattery, self-preference, sameness) persist as reasoning improves."),
        ("4", "Model identity matters: newest is not best on every dimension."),
        ("5", "Evaluate continuously, across model generations, with frozen tests."),
    ]
    for i, (n, text) in enumerate(items):
        y = 9.4 - i * 1.35
        round_box(ax, 0.2, y - 0.5, 1.4, 1.0, C_DARK, C_DARK, lw=0)
        ax.text(0.9, y, n, ha="center", va="center", color="white", fontsize=FS_HEAD,
                fontweight="bold")
        ax.text(2.1, y, text, va="center", fontsize=FS_HEAD + 1)
    round_box(ax, 0.2, 0.3, 23.6, 1.9, "#d6eaf8", C_DARK, lw=2)
    ax.text(12, 1.25, "Treat LLM performance as a multidimensional, time-dependent function,\n"
            "not a static property of a model family.", ha="center", va="center",
            fontsize=FS_HEAD + 1, fontweight="bold", color=C_DARK, linespacing=1.4)
    return fig


# -------------------------------------------------------
# Main
# -------------------------------------------------------
BUILDERS = [
    build_preface, build_step_1, build_step_2, *DIM_BUILDERS, build_step_11, build_step_12,
    build_step_13, build_step_14, build_step_15, build_step_16, build_step_16_1, build_step_17,
    build_step_18, build_step_19, build_step_20, build_step_21, build_step_22,
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM performance over time - analysis deck")
    parser.add_argument("--demo", action="store_true", help="print vectors, rates and DB output")
    parser.add_argument("--map", action="store_true", help="print the dimension map")
    args = parser.parse_args()
    if args.map:
        print_map()
    elif args.demo:
        print_demo()
    else:
        base.run_slide_deck(BUILDERS)
