"""
9_15_LLM_findings_Claude.py

The shelf life of LLM findings: which aspects of LLM performance get better
over time, which do not, and how fast.

Reading: Owen Queen & James Zou (Stanford), "The shelf life of LLM findings",
  November 2, 2026.  https://zou-group.github.io/llm-claims-shelf-life/
  (read 2026-11-05; all claim data below was taken from that page)

The study took 50 widely cited published claims about LLMs (e.g. "ChatGPT
fabricates most of its citations", "facts in the middle of a long context get
lost", "LLM judges prefer their own answers"), re-ran each claim's experiment on
20 models released between July 2024 and September 2026 by 8 developers, and
fitted a curve to how the effect changes with model release date.

Headline: by September 2026, 44% of the claims had expired, 34% had weakened,
16% persisted and 6% had strengthened. Capability failures fade fast; preferences
and judgment biases (sycophancy, self-preference, homogeneity) do not.

Slides (Step N = main idea, Step N.1 = details or code; at most two panels each):
  0. Preface                              8. Survival curve (Kaplan-Meier, live)
  1. A finding that expired: citations    9. Where every claim stands today
  2. Aspects of LLM performance tested   10. What improved: capability failures
  3. One scale for 50 metrics            11. What did not: preferences & judgment
  4. Fitting a claim over time (live)    12. Model-level view
  5. Three trajectories (live)           13. Why some aspects improve faster
  6. The verdicts                        14. What this means for research
  7. Lifetimes of findings               15. Takeaways

Part A of this file is a small, dependency-light (numpy only) re-implementation
of the study's method: the relative effect size e, the verdict rules, anchored
curve fits chosen by AICc with Theil-Sen outlier flagging and a bootstrap, the
expiration date, and the Kaplan-Meier lifetime estimate. The slides run this code
on the claim table and on per-model data for three claims (C11, C13, C28).

Usage:
  python 9_15_LLM_findings_Claude.py            # slide deck
  python 9_15_LLM_findings_Claude.py --demo     # summary + live fits in the terminal
  python 9_15_LLM_findings_Claude.py --claims   # print all 50 claims
"""

from __future__ import annotations

import argparse
import inspect
import random
import textwrap
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np


# =======================================================
# Part A: data
# =======================================================
@dataclass(frozen=True)
class Claim:
    cid: str
    title: str
    area: str
    cite: str
    published: str          # earliest public version (arXiv v1 etc.)
    verdict: str            # expired | weakened | persists | strengthened
    e_now: float            # fitted relative effect on 2026-09-30
    expired_on: str | None  # estimated expiration date (expired claims only)

    @property
    def pub(self) -> date:
        return date.fromisoformat(self.published)

    @property
    def death(self) -> date | None:
        return date.fromisoformat(self.expired_on) if self.expired_on else None


END_OF_PANEL = date(2026, 9, 30)   # last model release in the panel
VERDICTS = ["expired", "weakened", "persists", "strengthened"]

# Fitted values (e_now) and expiration dates as published by Queen & Zou (2026).
CLAIMS = [
    Claim('C1', 'LMs answer sub-questions yet fail the 2-hop question', 'Reasoning', 'Press et al. (2022)', '2022-10-07', 'weakened', 0.316, None),
    Claim('C2', '"Ignore previous instructions" hijacks prompted apps', 'Jailbreaks & safety robustness', 'Perez & Ribeiro (2022)', '2022-11-17', 'weakened', 0.247, None),
    Claim('C3', 'Territorial-dispute answers depend on query language', 'Social bias, culture & simulated populations', 'Li et al. (2023)', '2023-05-24', 'weakened', 0.307, None),
    Claim('C4', 'LLM judges flip verdicts when answer order is swapped', 'LLM judges & evaluation validity', 'Wang et al. (2023)', '2023-05-29', 'weakened', 0.53, None),
    Claim('C5', 'LLM personas of Black women stress "resilience"', 'Social bias, culture & simulated populations', 'Cheng et al. (2023)', '2023-05-29', 'expired', -0.024, '2026-06-15'),
    Claim('C6', 'LLMs add well in base 10 but fail in base 9', 'Reasoning', 'Wu et al. (2023)', '2023-07-05', 'expired', 0.025, '2024-01-10'),
    Claim('C7', 'Long-context QA is U-shaped: facts in the middle are lost', 'Long context, agents & tool use', 'Liu et al. (2024)', '2023-07-06', 'expired', 0.065, '2024-09-27'),
    Claim('C8', 'Safety-tuned models refuse clearly safe prompts', 'Jailbreaks & safety robustness', 'Röttger et al. (2023)', '2023-08-02', 'expired', 0.081, '2024-10-17'),
    Claim('C9', 'LLMs resolve ambiguous pronouns by gender stereotype', 'Social bias, culture & simulated populations', 'Kotek et al. (2023)', '2023-08-28', 'weakened', 0.711, None),
    Claim('C10', 'RAG models answer instead of rejecting answerless docs', 'Factuality, hallucination & calibration', 'Chen et al. (2023)', '2023-09-04', 'persists', 1.289, None),
    Claim('C11', 'ChatGPT fabricates most scholarly citations it writes', 'Science, research ideas & peer review', 'Walters & Wilder (2023)', '2023-09-07', 'expired', 0.031, '2026-04-29'),
    Claim('C12', 'LLM MCQ accuracy depends on which letter holds the answer', 'LLM judges & evaluation validity', 'Zheng et al. (2023)', '2023-09-07', 'expired', 0.018, '2024-04-24'),
    Claim('C13', 'LLMs answer "Who is X\'s parent?" but not the reverse', 'Reasoning', 'Berglund et al. (2023)', '2023-09-21', 'weakened', 0.527, None),
    Claim('C14', 'LLMs decode high-probability text better than rare text', 'Reasoning', 'McCoy et al. (2023)', '2023-09-24', 'weakened', 0.472, None),
    Claim('C15', 'Self-correction without feedback lowers reasoning accuracy', 'Reasoning', 'Huang et al. (2023)', '2023-10-03', 'expired', 0.178, '2024-11-09'),
    Claim('C16', 'Assistants abandon correct answers when asked "Are you sure?"', 'Sycophancy, deception & strategic behavior', 'Sharma et al. (2023)', '2023-10-20', 'expired', 0.077, '2024-09-24'),
    Claim('C17', 'Assistants praise text more when the user likes it', 'Sycophancy, deception & strategic behavior', 'Sharma et al. (2023)', '2023-10-20', 'strengthened', 1.846, None),
    Claim('C18', 'GPT-4 trading agent acts on insider tips under pressure', 'Sycophancy, deception & strategic behavior', 'Scheurer et al. (2023)', '2023-11-09', 'expired', 0.04, '2024-08-25'),
    Claim('C19', 'Persuasive multi-turn dialogue flips correct answers', 'Sycophancy, deception & strategic behavior', 'Xu et al. (2023)', '2023-12-14', 'expired', 0.083, '2026-02-08'),
    Claim('C20', 'LLMs hallucinate most answers about US federal cases', 'Factuality, hallucination & calibration', 'Dahl et al. (2024)', '2024-01-02', 'weakened', 0.727, None),
    Claim('C21', 'LLMs lean left on political orientation tests', 'Social bias, culture & simulated populations', 'Rozado (2024)', '2024-02-02', 'persists', 0.99, None),
    Claim('C22', 'Shuffling premise order breaks LLM logical proofs', 'Reasoning', 'Chen et al. (2024)', '2024-02-14', 'expired', 0.258, '2026-09-22'),
    Claim('C23', 'Fake references sway LLM judges toward the cited answer', 'LLM judges & evaluation validity', 'Chen et al. (2024)', '2024-02-16', 'expired', -0.101, '2025-10-01'),
    Claim('C24', 'LLMs convict African American English speakers more', 'Social bias, culture & simulated populations', 'Hofmann et al. (2024)', '2024-03-01', 'expired', -0.221, '2024-11-09'),
    Claim('C25', '"IMPORTANT!!! Ignore..." in tool output raises agent ASR', 'Jailbreaks & safety robustness', 'Zhan et al. (2024)', '2024-03-05', 'expired', -0.02, '2024-05-11'),
    Claim('C26', 'Many-shot prompts jailbreak models as shot count grows', 'Jailbreaks & safety robustness', 'Anil et al. (2024)', '2024-04-02', 'weakened', 0.625, None),
    Claim('C27', 'LLM judges prefer longer answers', 'LLM judges & evaluation validity', 'Dubois et al. (2024)', '2024-04-06', 'weakened', 0.65, None),
    Claim('C28', 'LLM evaluators prefer their own summaries', 'LLM judges & evaluation validity', 'Panickssery et al. (2024)', '2024-04-15', 'strengthened', 2.364, None),
    Claim('C29', 'LLMs asked for a random number are far from uniform', 'Creativity, output diversity & homogeneity', 'Zhang et al. (2024)', '2024-04-16', 'strengthened', 1.696, None),
    Claim('C30', 'Code LLMs recommend packages that do not exist', 'Factuality, hallucination & calibration', 'Spracklen et al. (2024)', '2024-06-12', 'weakened', 0.309, None),
    Claim('C31', 'LLM reviewers score accepted and rejected papers alike', 'Science, research ideas & peer review', 'Du et al. (2024)', '2024-06-24', 'persists', 0.916, None),
    Claim('C32', 'ChatGPT refuses more when user persona opposes request', 'Sycophancy, deception & strategic behavior', 'Li et al. (2024)', '2024-07-09', 'expired', 0.196, '2026-01-08'),
    Claim('C33', 'LLMs comply with illogical drug-switch requests', 'Sycophancy, deception & strategic behavior', 'Chen et al. (2024)', '2024-09-30', 'expired', 0.116, '2025-07-02'),
    Claim('C34', 'Hidden prompt injection steers LLM peer reviews', 'Science, research ideas & peer review', 'Ye et al. (2024)', '2024-12-02', 'weakened', 0.502, None),
    Claim('C35', 'Chat LLMs say "strawberry" has 2 r\'s, not 3', 'Reasoning', 'Fu et al. (2024)', '2024-12-19', 'expired', 0.023, '2026-04-29'),
    Claim('C36', 'Reasoning models overthink easy math', 'Reasoning', 'Chen et al. (2024)', '2024-12-30', 'expired', -0.072, '2025-02-12'),
    Claim('C37', 'Spin in trial abstracts inflates LLM benefit ratings', 'Science, research ideas & peer review', 'Yun et al. (2025)', '2025-02-11', 'persists', 0.828, None),
    Claim('C38', 'Models state falsehoods under pressure despite knowing truth', 'Sycophancy, deception & strategic behavior', 'Ren et al. (2025)', '2025-03-05', 'weakened', 0.518, None),
    Claim('C39', 'LLMs get lost when instructions arrive over many turns', 'Long context, agents & tool use', 'Laban et al. (2025)', '2025-05-09', 'weakened', 0.278, None),
    Claim('C40', 'LLMs side with posters the crowd judged at fault (AITA)', 'Sycophancy, deception & strategic behavior', 'Cheng et al. (2025)', '2025-05-20', 'weakened', 0.603, None),
    Claim('C41', 'LLMs fail to abstain on unanswerable questions', 'Factuality, hallucination & calibration', 'Kirichenko et al. (2025)', '2025-06-10', 'persists', 1.161, None),
    Claim('C42', "AI affirms users' actions more often than humans do", 'Sycophancy, deception & strategic behavior', 'Cheng et al. (2026)', '2025-10-01', 'persists', 1.057, None),
    Claim('C43', 'AI endorses problematic actions users describe', 'Sycophancy, deception & strategic behavior', 'Cheng et al. (2026)', '2025-10-01', 'persists', 1.125, None),
    Claim('C44', 'Frontier models blackmail an executive to avoid replacement', 'Sycophancy, deception & strategic behavior', 'Lynch et al. (2025)', '2025-06-20', 'expired', 0.174, '2026-09-09'),
    Claim('C45', 'LLMs give near-identical answers to open-ended prompts', 'Creativity, output diversity & homogeneity', 'Jiang et al. (2025)', '2025-10-27', 'persists', 1.0, None),
    Claim('C46', 'LLMs leak information marked private when summarizing meetings', 'Jailbreaks & safety robustness', 'Mireshghallah et al. (2023)', '2023-10-27', 'expired', 0.059, '2025-09-08'),
    Claim('C47', 'LLMs give contradictory answers to paired before/after timeline questions', 'Reasoning', 'Qiu et al. (2023)', '2023-11-14', 'weakened', 0.426, None),
    Claim('C48', 'LLMs show implicit stereotype bias in IAT-style word association', 'Social bias, culture & simulated populations', 'Bai et al. (2024)', '2024-02-06', 'weakened', 0.246, None),
    Claim('C49', 'LLMs prefer AI-written research abstracts over human-written ones', 'LLM judges & evaluation validity', 'Laurito et al. (2024)', '2024-07-09', 'expired', -0.589, '2025-09-28'),
    Claim('C50', 'LLMs give more nonanswers to political-biography requests in Arabic than in English', 'Social bias, culture & simulated populations', 'Noels et al. (2025)', '2025-04-04', 'expired', 0.323, '2026-09-22'),
]
CLAIM = {c.cid: c for c in CLAIMS}

# Short names for the eight research areas (= the "aspects of LLM performance").
AREA_SHORT = {
    "Reasoning": "Reasoning",
    "Long context, agents & tool use": "Long context & agents",
    "Factuality, hallucination & calibration": "Factuality & hallucination",
    "Science, research ideas & peer review": "Science & peer review",
    "Jailbreaks & safety robustness": "Jailbreaks & safety",
    "Sycophancy, deception & strategic behavior": "Sycophancy & deception",
    "LLM judges & evaluation validity": "LLMs as judges",
    "Social bias, culture & simulated populations": "Social bias & culture",
    "Creativity, output diversity & homogeneity": "Output diversity",
}

# The 20-model panel (release dates), oldest first.
PANEL = [
    ("GPT-4o mini", "2024-07-18"), ("Llama 3.1 70B Instruct", "2024-07-23"),
    ("Qwen2.5-72B-Instruct", "2024-09-19"), ("DeepSeek V3", "2024-12-26"),
    ("Mistral Small 3 (2501)", "2025-01-30"), ("Llama 4 Maverick", "2025-04-05"),
    ("Gemini 2.5 Flash", "2025-05-20"), ("gpt-oss-120b", "2025-08-05"),
    ("Claude Haiku 4.5", "2025-10-15"), ("GPT-5.2", "2025-12-11"),
    ("GLM-5", "2026-02-12"), ("Claude Opus 4.6", "2026-02-05"),
    ("Qwen3.7-Max", "2026-05-20"), ("Qwen3.7-Flash", "2026-07-21"),
    ("DeepSeek V4 Pro 0813", "2026-08-13"), ("GLM-5.3-Flash", "2026-08-26"),
    ("Gemini 3.8 Flash", "2026-09-02"), ("GPT-6 Luna", "2026-09-22"),
    ("Claude Sonnet 5.5", "2026-09-28"), ("GPT-6.1 Sol", "2026-09-29"),
]

# Per-model relative effects for three claims (same order as PANEL), plus the
# reproduction anchor on the paper's model: (date, e, 95% low, 95% high) and the
# fitted result the authors published, for comparison with our re-fit.
PANEL_E = {
    "C11": dict(
        e=[1.2333, 0.6671, 0.869, 0.3626, 0.5708, 0.4818, 0.7392, 1.1673, 0.4326, 0.0942,
           0.3727, 0.0783, 0.1976, 1.2456, 0.0868, 0.1473, 0.309, 0.0507, 0.0249, 0.0],
        anchor=("2024-01-25", 1.0655, 0.8673, 1.2582), anchor_model="gpt-3.5-turbo-0125",
        metric="share of fabricated references (paper: 55%)",
        published_fit="linear, e = 0.031 [-0.076, 0.121], expired 2026-04-29"),
    "C13": dict(
        e=[1.183, 0.9872, 0.5855, 1.3867, 1.1419, 1.38, 1.5621, 0.4928, 1.2214, 1.7447,
           0.5997, 0.7764, 0.6617, 0.5258, 0.5115, 0.5723, 0.593, 0.6155, 0.5326, 0.4057],
        anchor=("2024-01-25", 1.0905, 0.8499, 1.3127), anchor_model="gpt-3.5-turbo-0125",
        metric="forward - reverse accuracy gap (paper: 43.7 pp)",
        unreliable={4, 5},
        published_fit="linear, e = 0.528 [0.441, 0.609], weakened"),
    "C28": dict(
        e=[0.2717, 0.7609, 0.3804, 0.2717, 1.4674, -0.7609, 1.0326, 1.7391, 0.163, 0.8696,
           2.4956, 0.3804, 2.5, 1.3043, 3.587, 2.1848, 4.5109, 1.6848, 1.087, 2.8804],
        anchor=("2024-04-09", 1.078, 0.712, 1.444), anchor_model="gpt-4-turbo-2024-04-09",
        metric="own - neutral judge preference (paper: 4.6 pp)",
        published_fit="exponential, e = 2.364 [1.251, 3.066], strengthened"),
}

# Model-level counts quoted in the study (claims at e <= 0.2 out of 50).
MODEL_EXPIRED_COUNTS = [
    ("GPT-4o mini (Jul 2024)", 5), ("DeepSeek V4 Pro", 11), ("Qwen3.7-Flash", 14),
    ("GLM-5.3-Flash", 19), ("GPT-6 Luna", 25), ("GPT-6.1 Sol (Sep 2026)", 30),
]


# =======================================================
# Part A: method
# =======================================================
def relative_effect(model_value, paper_value, expiration_value):
    """e = 1: as strong as in the paper; e = 0: the claimed effect is gone.

    The expiration value is the metric value at which the effect is absent
    (0 fabricated citations, 50% stereotype-congruent choices, uniform entropy...).
    If it lies above the paper value, the sign flips automatically.
    """
    return (model_value - expiration_value) / (paper_value - expiration_value)


def zone(e):
    """Thresholds on the relative-effect scale."""
    if e <= 0.2:
        return "expired"        # a fifth of the paper's effect, or less
    if e < 0.8:
        return "weakened"
    if e < 1.5:
        return "persists"       # at least 80% of the effect remains
    return "strengthened"


def years(d: date, t0: date) -> float:
    return (d - t0).days / 365.25


def months(d1: date, d0: date) -> float:
    return (d1 - d0).days / (365.25 / 12)


# --- candidate curves, all forced through the anchor (t = 0, e = e0) --------
def _grid_fit(t, y, e0, kind):
    """Least squares over a parameter grid; returns (rss, n_params, f(t_new))."""
    if kind == "constant":
        return float(np.sum((y - e0) ** 2)), 0, lambda s: np.full_like(np.asarray(s, float), e0)
    if kind == "linear":
        b = float(np.sum(t * (y - e0)) / max(np.sum(t * t), 1e-12))
        return float(np.sum((y - e0 - b * t) ** 2)), 1, lambda s: e0 + b * np.asarray(s, float)
    if kind == "exponential":
        rates = np.linspace(-4.0, 2.0, 241)                         # per year
        pred = e0 * np.exp(rates[:, None] * t[None, :])
        rss = np.sum((pred - y) ** 2, axis=1)
        r = rates[np.argmin(rss)]
        return float(rss.min()), 1, lambda s: e0 * np.exp(r * np.asarray(s, float))
    if kind == "logistic":                                          # S-curve to 0
        ks = np.concatenate([np.linspace(-12, -0.5, 24), np.linspace(0.5, 12, 24)])
        cs = np.linspace(0.0, 3.0, 61)                              # midpoint, years
        K, C = np.meshgrid(ks, cs, indexing="ij")
        K, C = K.ravel(), C.ravel()
        norm = 1 + np.exp(-K * C)
        pred = e0 * norm[:, None] / (1 + np.exp(K[:, None] * (t[None, :] - C[:, None])))
        rss = np.sum((pred - y) ** 2, axis=1)
        j = int(np.argmin(rss))
        k, c, nm = K[j], C[j], norm[j]
        return float(rss[j]), 2, lambda s: e0 * nm / (1 + np.exp(k * (np.asarray(s, float) - c)))
    raise ValueError(kind)


def aicc(rss, n, p):
    k = p + 1                                   # + noise variance
    return n * np.log(max(rss, 1e-12) / n) + 2 * k + 2 * k * (k + 1) / max(n - k - 1, 1)


def theil_sen_outliers(t, y, cap=0.2, z=3.0):
    """Flag points > z robust SDs from a Theil-Sen line, at most `cap` of them."""
    i, j = np.triu_indices(len(t), 1)
    ok = t[j] != t[i]
    slope = np.median((y[j] - y[i])[ok] / (t[j] - t[i])[ok])
    resid = y - (np.median(y - slope * t) + slope * t)
    sd = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    far = np.where(np.abs(resid) > z * max(sd, 1e-9))[0]
    far = far[np.argsort(-np.abs(resid[far]))][: int(cap * len(t))]
    mask = np.zeros(len(t), bool)
    mask[far] = True
    return mask


def fit_claim(t, y, e0):
    """Fit constant / linear / exponential / logistic through the anchor; best AICc."""
    y = np.clip(y, -1, 3)                       # one extreme model cannot dominate
    best = None
    for kind in ("constant", "linear", "exponential", "logistic"):
        rss, p, f = _grid_fit(t, y, e0, kind)
        score = aicc(rss, len(t), p)
        if best is None or score < best[0]:
            best = (score, kind, f)
    return best[1], best[2]


def expiration_time(f, t_end, step=1 / 365.25):
    """First t from which the curve stays at e <= 0.2 through t_end, else None."""
    s = np.arange(0, t_end + step, step)
    above = np.where(f(s) > 0.2)[0]
    if len(above) == 0:
        return 0.0
    last = above[-1]
    return None if last == len(s) - 1 else float(s[last + 1])


def analyze_claim(cid, n_boot=300, seed=0):
    """Re-run the study's trend fit for one claim from its per-model effects."""
    d = PANEL_E[cid]
    t0 = date.fromisoformat(d["anchor"][0])
    e0, lo, hi = d["anchor"][1:]
    keep = [i for i, (_, rel) in enumerate(PANEL)
            if date.fromisoformat(rel) >= t0 and i not in d.get("unreliable", set())]
    t = np.array([years(date.fromisoformat(PANEL[i][1]), t0) for i in keep])
    y = np.array([d["e"][i] for i in keep], float)
    out = theil_sen_outliers(t, y)
    t_fit, y_fit = t[~out], y[~out]
    t_end = years(END_OF_PANEL, t0)

    kind, f = fit_claim(t_fit, y_fit, e0)
    e_end = float(f(t_end))
    t_exp = expiration_time(f, t_end)

    rng = random.Random(seed)       # stdlib RNG (numpy.random is blocked on some lab PCs)
    sd0 = (hi - lo) / (2 * 1.96)
    grid = np.linspace(0, t_end, 120)
    boot_end, boot_curves = [], []
    for _ in range(n_boot):
        idx = np.array([rng.randrange(len(t_fit)) for _ in range(len(t_fit))])
        _, fb = fit_claim(t_fit[idx], y_fit[idx], rng.gauss(e0, sd0))
        boot_end.append(float(fb(t_end)))
        boot_curves.append(fb(grid))
    band = np.percentile(np.array(boot_curves), [2.5, 97.5], axis=0)
    ci = tuple(np.percentile(boot_end, [2.5, 97.5]))
    return dict(cid=cid, t0=t0, e0=e0, anchor_ci=(lo, hi), t=t, y=y, outlier=out, kind=kind,
                f=f, e_end=e_end, ci=ci, t_end=t_end, grid=grid, band=band,
                expired_on=(t0 + timedelta(days=round(t_exp * 365.25))) if t_exp is not None else None,
                zone=zone(e_end), names=[PANEL[i][0] for i in keep])


def kaplan_meier(claims=CLAIMS, end=END_OF_PANEL):
    """Share of claims expired vs months since publication (censored at `end`)."""
    obs = sorted(
        (months(c.death, c.pub), True) if c.death else (months(end, c.pub), False)
        for c in claims)
    n_at_risk, s, curve = len(obs), 1.0, [(0.0, 0.0)]
    for m, died in obs:
        if died:
            s *= 1 - 1 / n_at_risk
            curve.append((m, 1 - s))
        n_at_risk -= 1
    return curve


def share_expired_by(curve, m):
    return max(v for t, v in curve if t <= m)


def lifetimes(claims=CLAIMS):
    return [months(c.death, c.pub) for c in claims if c.death]


def verdict_counts(claims=CLAIMS):
    return Counter(c.verdict for c in claims)


def area_table(claims=CLAIMS):
    rows = {}
    for c in claims:
        rows.setdefault(c.area, Counter())[c.verdict] += 1
    return rows


# =======================================================
# Terminal views
# =======================================================
def print_claims():
    for c in CLAIMS:
        exp = f"expired {c.expired_on}" if c.expired_on else ""
        print(f"{c.cid:>4}  {c.verdict:<12} e={c.e_now:6.2f}  {c.published}  "
              f"{AREA_SHORT[c.area]:<27} {c.title}  [{c.cite}] {exp}")


def print_demo():
    vc = verdict_counts()
    print("Verdicts on 50 claims (Sep 2026):",
          ", ".join(f"{v} {vc[v]} ({vc[v] * 2}%)" for v in VERDICTS))
    print("\nBy aspect (research area):")
    for area, cnt in sorted(area_table().items(), key=lambda kv: -sum(kv[1].values())):
        print(f"  {AREA_SHORT[area]:<28}" + "  ".join(f"{v[:4]} {cnt[v]}" for v in VERDICTS))
    km = kaplan_meier()
    print("\nKaplan-Meier share expired: "
          + ", ".join(f"{m} mo: {share_expired_by(km, m):.0%}" for m in (6, 12, 24)))
    lt = lifetimes()
    print(f"Expired claims: mean lifetime {np.mean(lt):.1f} months, median {np.median(lt):.1f}")
    print("\nRe-fitting three claims from per-model data:")
    for cid in PANEL_E:
        r = analyze_claim(cid)
        print(f"  {cid} {CLAIM[cid].title}")
        print(f"     ours: {r['kind']}, e(2026-09-30) = {r['e_end']:.3f} "
              f"[{r['ci'][0]:.3f}, {r['ci'][1]:.3f}] -> {r['zone']}"
              + (f", crosses 0.2 on {r['expired_on']}" if r['expired_on'] else ""))
        print(f"     paper: {PANEL_E[cid]['published_fit']}")


# =======================================================
# Part B: slides
# =======================================================
# Readability rules (lecture screen ~10.8 x 6.1 in): <= 2 panels per slide,
# body text >= 12 pt, notes >= 11 pt, code >= 10 pt.
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from _config import (  # noqa: E402
    FS_FIGURE_TITLE,
    FS_TICK,
    FS_ANNOT,
    FIGURE_DPI,
    SCREEN_FRACTION,
    apply_rcparams,
    run_slide_deck,
)

apply_rcparams(plt)

FS_TITLE = FS_FIGURE_TITLE - 2
FS_HEAD = FS_ANNOT
FS_BODY = FS_TICK - 2
FS_NOTE = FS_TICK - 3
FS_CODE_MAX, FS_CODE_MIN = FS_TICK - 1, FS_TICK - 4

C_DARK = "#1a5276"
C_MUTE = "#f4f6f7"
C_CODE = "#1e1e1e"
C_ACCENT = "#c0392b"
C_OK = "#1e8449"

V_COLOR = {"expired": "#7f8c8d", "weakened": "#e67e22", "persists": "#2471a3",
           "strengthened": "#c0392b"}
V_FACE = {"expired": "#eaeded", "weakened": "#fdebd0", "persists": "#d6eaf8",
          "strengthened": "#fadbd8"}


def screen_figsize(fraction=SCREEN_FRACTION, dpi=FIGURE_DPI):
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return (w / dpi * fraction, h / dpi * fraction)
    except Exception:
        return (16 * fraction / 0.95, 9 * fraction / 0.95)


def slide(title, ncols=1, nrows=1, **gs_kw):
    fig = plt.figure(figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig.suptitle(title, fontsize=FS_TITLE, fontweight="bold", y=0.965, color=C_DARK)
    for k, v in dict(top=0.85, bottom=0.19, left=0.08, right=0.97, wspace=0.25).items():
        gs_kw.setdefault(k, v)
    return fig, fig.add_gridspec(nrows, ncols, **gs_kw)


def blank(ax, xlim=(0, 12), ylim=(0, 10)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")


def style(ax, title=None, xlabel=None, ylabel=None):
    ax.tick_params(labelsize=FS_BODY)
    if title:
        ax.set_title(title, fontsize=FS_BODY + 1, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FS_BODY)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FS_BODY)


def round_box(ax, x, y, w, h, face, edge="0.2", lw=2.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                facecolor=face, edgecolor=edge, linewidth=lw))


def card(ax, x, y, w, h, title, body, face, edge):
    round_box(ax, x, y, w, h, face, edge, lw=1.8)
    ax.text(x + w / 2, y + h - 0.35, title, ha="center", va="top", fontsize=FS_HEAD,
            fontweight="bold", color=edge, parse_math=False)
    ax.text(x + w / 2, y + h - 1.2, body, ha="center", va="top", fontsize=FS_BODY,
            color="0.15", linespacing=1.4, parse_math=False)


def text_block(ax, x, y, text, fs=None, **kw):
    kw.setdefault("ha", "left")
    kw.setdefault("va", "top")
    kw.setdefault("linespacing", 1.45)
    ax.text(x, y, text, fontsize=fs or FS_BODY, parse_math=False, **kw)


def row_list(ax, items, top, step, x=0.2, w=11.6, face=C_MUTE, edge="0.6"):
    h = step * 0.82
    for i, (head, sub) in enumerate(items):
        y = top - i * step
        round_box(ax, x, y - h, w, h, face, edge, lw=1.2)
        ax.text(x + 0.3, y - h * 0.32, head, ha="left", va="center", fontsize=FS_BODY,
                fontweight="bold", color="0.1", parse_math=False)
        ax.text(x + 0.3, y - h * 0.72, sub, ha="left", va="center", fontsize=FS_NOTE,
                color="0.25", parse_math=False)


def code_panel(ax, code, title=""):
    lines = code.rstrip("\n").splitlines() or [""]
    box, fig = ax.get_position(), ax.figure
    w_pt = 0.93 * box.width * fig.get_figwidth() * 72
    h_pt = 0.9 * box.height * fig.get_figheight() * 72
    fs = max(FS_CODE_MIN, min(FS_CODE_MAX, w_pt / (0.61 * max(map(len, lines))),
                              h_pt / (1.3 * 1.2 * (len(lines) + (2 if title else 0)))))
    blank(ax, xlim=(0, 1), ylim=(0, 1))
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.0,rounding_size=0.02",
                                facecolor=C_CODE, edgecolor="#111111", lw=1.2,
                                transform=ax.transAxes, clip_on=False))
    top = 0.97
    if title:
        ax.text(0.025, top, title, ha="left", va="top", fontsize=FS_BODY, fontweight="bold",
                color="#9cdcfe", transform=ax.transAxes, parse_math=False)
        top -= FS_BODY * 2.0 / (box.height * fig.get_figheight() * 72)
    ax.text(0.025, top, "\n".join(lines), ha="left", va="top", family="monospace",
            fontsize=fs, color="#d4d4d4", linespacing=1.3, transform=ax.transAxes,
            parse_math=False)


def footnote(fig, text, y=0.078):
    fig.text(0.525, y, text, ha="center", va="center", fontsize=FS_NOTE, color="0.35",
             style="italic", parse_math=False)


def source_of(fn):
    return textwrap.dedent(inspect.getsource(fn))


def verdict_legend(ax, loc="lower right"):
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=V_COLOR[v], label=v) for v in VERDICTS], loc=loc,
              fontsize=FS_NOTE, frameon=False)


def date_axis(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def to_dt(d):
    return np.datetime64(d.isoformat()) if isinstance(d, date) else np.datetime64(d)


_CACHE: dict = {}


def analysis(cid):
    if cid not in _CACHE:
        _CACHE[cid] = analyze_claim(cid)
    return _CACHE[cid]


# -------------------------------------------------------
# Preface
# -------------------------------------------------------
def build_preface():
    fig, gs = slide("")
    ax = fig.add_subplot(gs[0, 0])
    blank(ax)
    ax.text(6, 9.4, "Reading", ha="center", va="center", fontsize=FS_HEAD + 2,
            fontweight="bold", color="0.45")
    round_box(ax, 0.6, 5.9, 10.8, 2.8, "#d6eaf8", C_DARK, lw=2.2)
    ax.text(6, 7.65, "The shelf life of LLM findings", ha="center", va="center",
            fontsize=FS_FIGURE_TITLE + 10, fontweight="bold", color=C_DARK)
    ax.text(6, 6.55, "Which aspects of LLM performance improve over time, and how fast?",
            ha="center", va="center", fontsize=FS_HEAD + 3, color="0.25")
    ax.text(6, 4.9, "Owen Queen & James Zou  ·  Stanford  ·  November 2, 2026",
            ha="center", va="center", fontsize=FS_HEAD + 4, fontweight="bold")
    ax.text(6, 3.75, "50 published claims  ·  20 models (Jul 2024 - Sep 2026)  ·  8 developers",
            ha="center", va="center", fontsize=FS_HEAD + 1, color="0.3")
    ax.text(6, 2.2, "zou-group.github.io/llm-claims-shelf-life", ha="center", va="center",
            fontsize=FS_HEAD + 2, fontweight="bold", color=C_DARK)
    ax.text(6, 1.3, "read 2026-11-05  ·  numbers on these slides come from that page or are "
            "re-computed from its data", ha="center", va="center", fontsize=FS_BODY, color="0.45")
    return fig


# -------------------------------------------------------
# Step 1: a finding that expired
# -------------------------------------------------------
def build_step_1():
    fig, gs = slide("Step 1: A finding that expired - fabricated citations (C11)", ncols=2,
                    width_ratios=[1, 1.1], wspace=0.15)
    ax = fig.add_subplot(gs[0, 0])
    blank(ax)
    text_block(ax, 0.2, 9.6,
               "2023: Walters & Wilder ask ChatGPT (GPT-3.5)\n"
               "for short literature reviews and look up\n"
               "every reference.\n\n"
               "  55% of the cited works did not exist.\n"
               "  73% of cited journal articles were invented.\n"
               "  Only 2% named a fake journal: the fakes\n"
               "  looked plausible.\n\n"
               "2026: same prompts, 42 topics, today's models.",
               fs=FS_HEAD)
    round_box(ax, 0.2, 0.1, 11.4, 2.3, "#fdebd0", "#e67e22", lw=1.6)
    text_block(ax, 0.5, 2.15, "Policies built on the old number:\n"
               "ICLR 2026 desk-rejected papers with hallucinated\n"
               "references; ICML 2026 lists them as grounds.", fs=FS_BODY)
    ax = fig.add_subplot(gs[0, 1])
    labels = ["GPT-3.5\npaper", "GPT-3.5T\nre-run", "GPT-4\npaper",
              "Qwen3.7\nFlash", "Sonnet\n5.5", "GPT-6.1\nSol"]
    vals = [55, 59, 18, 69, 1.4, 0]
    cols = ["#7f8c8d", "#7f8c8d", "#7f8c8d", C_ACCENT, C_OK, C_OK]
    bars = ax.bar(range(len(vals)), vals, color=cols)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:g}%", ha="center",
                fontsize=FS_BODY, fontweight="bold")
    ax.set_xticks(range(len(vals)), labels, fontsize=FS_NOTE)
    ax.set_ylim(0, 80)
    style(ax, "Share of references that are fabricated", ylabel="% fabricated")
    ax.text(4, 74, "2026 models", ha="center", fontsize=FS_NOTE, color="0.35")
    ax.plot([2.6, 5.4], [72, 72], color="0.6", lw=1)
    ax.spines[["top", "right"]].set_visible(False)
    footnote(fig, "Expired overall - but not for every model: one recent model still invents "
             "69% of its references.")
    return fig


# -------------------------------------------------------
# Step 2: aspects of LLM performance
# -------------------------------------------------------
def build_step_2():
    fig, gs = slide("Step 2: Nine aspects of LLM behaviour, 50 claims", left=0.24, right=0.95)
    ax = fig.add_subplot(gs[0, 0])
    rows = sorted(area_table().items(), key=lambda kv: sum(kv[1].values()))
    left = np.zeros(len(rows))
    for v in VERDICTS:
        vals = np.array([cnt[v] for _, cnt in rows])
        ax.barh(range(len(rows)), vals, left=left, color=V_COLOR[v], label=v, height=0.7)
        for i, (x, w) in enumerate(zip(left, vals)):
            if w:
                ax.text(x + w / 2, i, str(w), ha="center", va="center", color="white",
                        fontsize=FS_BODY, fontweight="bold")
        left += vals
    ax.set_yticks(range(len(rows)), [AREA_SHORT[a] for a, _ in rows], fontsize=FS_BODY)
    style(ax, xlabel="number of claims (verdict in September 2026)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=FS_NOTE, frameon=False, loc="lower right", ncol=2)
    footnote(fig, "From the claim table in Part A.  Reasoning: 5 of 9 expired, none persist.  "
             "Sycophancy: 6 of 11 expired, 3 persist or grow.")
    return fig


def build_step_2_1():
    fig, gs = slide("Step 2.1: What each aspect measures (example claims)")
    ax = fig.add_subplot(gs[0, 0])
    blank(ax)
    items = [
        ("Reasoning", "2-hop questions, reversal curse, base-9 arithmetic, 'strawberry' r's, premise order"),
        ("Long context, agents & tool use", "facts lost in the middle; getting lost over many turns"),
        ("Factuality & hallucination", "legal hallucinations, invented packages, failing to abstain"),
        ("Science & peer review", "fake citations, LLM reviewers, spin in abstracts, hidden injections"),
        ("Jailbreaks & safety", "prompt injection, many-shot jailbreaks, over-refusal, privacy leaks"),
        ("Sycophancy & deception", "'Are you sure?' flips, flattery, insider trading, blackmail"),
        ("LLMs as judges", "position bias, length bias, self-preference, fake-reference sway"),
        ("Social bias & culture / output diversity", "dialect bias, political lean, language-dependent answers; "
         "random numbers, homogeneous answers"),
    ]
    row_list(ax, items, top=9.9, step=1.22)
    return fig


# -------------------------------------------------------
# Step 3: one scale for 50 metrics
# -------------------------------------------------------
def build_step_3():
    fig, gs = slide("Step 3: Putting 50 different metrics on one scale", ncols=2,
                    width_ratios=[1, 1], wspace=0.1)
    ax = fig.add_subplot(gs[0, 0])
    blank(ax)
    text_block(ax, 0.2, 9.6, "Relative effect size", fs=FS_HEAD + 2, fontweight="bold")
    text_block(ax, 0.2, 8.6, "e = (model value - expiration value)\n"
               "     / (paper value - expiration value)", fs=FS_HEAD + 1, family="monospace")
    text_block(ax, 0.2, 6.7,
               "e = 1   as strong as in the paper\n"
               "e = 0   the claimed effect is gone\n\n"
               "Expiration value = metric with no effect:\n"
               "  0 fabricated citations (C11)\n"
               "  50% stereotype-congruent choices (C9)\n"
               "  entropy of a uniform draw (C29)\n\n"
               "C11, Claude Sonnet 5.5: 1.37% fake\n"
               f"  e = (0.0137 - 0) / (0.55 - 0) = {relative_effect(0.0137, 0.55, 0):.3f}",
               fs=FS_BODY)
    ax = fig.add_subplot(gs[0, 1])
    blank(ax, xlim=(0, 12), ylim=(0, 10))
    bands = [("expired", "e ≤ 0.2", "≤ 1/5 of effect"),
             ("weakened", "0.2 < e < 0.8", "smaller"),
             ("persists", "0.8 ≤ e < 1.5", "≥ 80% remains"),
             ("strengthened", "e ≥ 1.5", "stronger")]
    for i, (v, lab, sub) in enumerate(bands):
        y = 8.9 - i * 1.35
        round_box(ax, 0.2, y - 0.55, 11.4, 1.1, V_FACE[v], V_COLOR[v], lw=1.8)
        ax.text(0.5, y, v, va="center", fontsize=FS_BODY, fontweight="bold", color=V_COLOR[v])
        ax.text(4.0, y, lab, va="center", fontsize=FS_BODY, family="monospace")
        ax.text(8.0, y, sub, va="center", fontsize=FS_NOTE, color="0.3")
    text_block(ax, 0.2, 3.3,
               "Verdicts also need the 95% bootstrap interval:\n"
               "persists / strengthened only if its lower end > 0.2.\n"
               "Expired: fitted curve reaches 0.2 and stays there, or\n"
               "≥ 4 of the 8 newest models are at e ≤ 0.2.",
               fs=FS_NOTE, color="0.25")
    return fig


def build_step_3_1():
    fig, gs = slide("Step 3.1: The scale and the zones in code (Part A)", top=0.87, bottom=0.13)
    ax = fig.add_subplot(gs[0, 0])
    code_panel(ax, source_of(relative_effect) + "\n" + source_of(zone),
               title="relative_effect() and zone()")
    return fig


# -------------------------------------------------------
# Step 4: fitting a claim over time
# -------------------------------------------------------
def plot_fit(ax, r, title):
    t0 = r["t0"]
    to_d = lambda s: [t0 + timedelta(days=float(x) * 365.25) for x in np.atleast_1d(s)]  # noqa: E731
    gx = to_d(r["grid"])
    ax.axhspan(-1.2, 0.2, color="#eaeded", zorder=0)
    ax.text(gx[1], 0.05, "expiration zone (e ≤ 0.2)", fontsize=FS_NOTE, color="0.35", va="center")
    ax.axhline(1, color="0.4", ls="--", lw=1)
    ax.fill_between(gx, r["band"][0], r["band"][1], color=V_COLOR[r["zone"]], alpha=0.18, lw=0)
    ax.plot(gx, r["f"](r["grid"]), color=V_COLOR[r["zone"]], lw=3,
            label=f"{r['kind']} fit (lowest AICc)")
    pd = to_d(r["t"])
    ok = ~r["outlier"]
    ax.scatter([p for p, k in zip(pd, ok) if k], r["y"][ok], s=45, color="0.2", zorder=3,
               label="panel model")
    if (~ok).any():
        ax.scatter([p for p, k in zip(pd, ok) if not k], r["y"][~ok], s=45, facecolor="none",
                   edgecolor="0.2", zorder=3, label="outlier (Theil-Sen)")
    lo, hi = r["anchor_ci"]
    ax.errorbar([t0], [r["e0"]], yerr=[[r["e0"] - lo], [hi - r["e0"]]], fmt="D", color="k",
                ms=9, capsize=4, zorder=4, label="reproduction on paper's model")
    if r["expired_on"]:
        ax.axvline(r["expired_on"], color=C_ACCENT, lw=1.5, ls=":")
        ax.text(r["expired_on"], 0.75, f"expires \n{r['expired_on']} ", ha="right",
                color=C_ACCENT, fontsize=FS_NOTE, va="center", fontweight="bold")
    ax.set_ylim(min(-0.3, r["y"].min() - 0.2), max(1.6, min(r["y"].max(), 3) + 0.3))
    date_axis(ax)
    style(ax, title, xlabel="model release date", ylabel="relative effect e")
    ax.spines[["top", "right"]].set_visible(False)


def build_step_4():
    fig, gs = slide("Step 4: Fitting how a claim changes with release date (C11, live)",
                    ncols=2, width_ratios=[1.7, 1], wspace=0.08)
    r = analysis("C11")
    ax = fig.add_subplot(gs[0, 0])
    plot_fit(ax, r, "Fabricated citations, 20 models")
    ax.set_ylim(-0.3, 2.1)
    ax.legend(fontsize=FS_NOTE, frameon=False, loc="upper right", ncol=2)
    ax = fig.add_subplot(gs[0, 1])
    blank(ax)
    text_block(ax, 0.3, 9.6,
               "1. Anchor: re-run the paper on its\n   model (or the closest one).\n"
               "2. Flag outliers with a Theil-Sen line\n   (> 3 robust SD, at most 20%).\n"
               "3. Fit constant / linear / exponential\n   / logistic, all through the anchor;\n"
               "   keep the lowest AICc.\n"
               "4. Bootstrap the models (band).\n"
               "5. Read e at 2026-09-30 and the\n   date it enters e ≤ 0.2.",
               fs=FS_BODY)
    round_box(ax, 0.3, 0.2, 11.4, 2.4, C_MUTE, "0.6", lw=1.2)
    text_block(ax, 0.6, 2.4,
               f"ours:  {r['kind']}, e = {r['e_end']:.3f}\n"
               f"       expires {r['expired_on']}\n"
               f"paper: linear, e = 0.031\n"
               f"       expires 2026-04-29",
               fs=FS_NOTE, family="monospace")
    footnote(fig, "Single models are noisy and disagree; the verdict comes from the trend, "
             "not from any one model.")
    return fig


def build_step_4_1():
    fig, gs = slide("Step 4.1: Choosing the curve shape by AICc (Part A)", top=0.87, bottom=0.13)
    code_panel(fig.add_subplot(gs[0, 0]), source_of(fit_claim) + "\n" + source_of(aicc),
               title="fit_claim() and aicc()")
    return fig


def build_step_4_2():
    fig, gs = slide("Step 4.2: Outliers and the expiration date (Part A)", top=0.87, bottom=0.13)
    code_panel(fig.add_subplot(gs[0, 0]), source_of(theil_sen_outliers)
               + "\n" + source_of(expiration_time),
               title="theil_sen_outliers() and expiration_time()")
    return fig


# -------------------------------------------------------
# Step 5: three trajectories
# -------------------------------------------------------
def build_step_5():
    fig, gs = slide("Step 5: Not every effect fades - two more trajectories (live)", ncols=2,
                    wspace=0.18)
    for k, cid in enumerate(("C13", "C28")):
        r = analysis(cid)
        ax = fig.add_subplot(gs[0, k])
        plot_fit(ax, r, f"{cid}: {CLAIM[cid].title}")
        ax.title.set_fontsize(FS_NOTE + 1)
        ax.text(0.02, 0.97, f"{r['zone']}: e = {r['e_end']:.2f} "
                f"[{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]", transform=ax.transAxes, va="top",
                fontsize=FS_BODY, fontweight="bold", color=V_COLOR[r["zone"]])
    footnote(fig, "Reversal curse: about half remains.  Self-preference: grows, but from a "
             "small 4.6 pp reference and with a wide spread.")
    return fig


# -------------------------------------------------------
# Step 6: verdicts
# -------------------------------------------------------
def build_step_6():
    fig, gs = slide("Step 6: The verdicts - most findings faded", ncols=2,
                    width_ratios=[1, 1.1], wspace=0.25)
    vc = verdict_counts()
    ax = fig.add_subplot(gs[0, 0])
    ax.pie([vc[v] for v in VERDICTS], colors=[V_COLOR[v] for v in VERDICTS], startangle=90,
           counterclock=False, wedgeprops=dict(width=0.42, edgecolor="white"),
           labels=[f"{v}\n{vc[v]} ({vc[v] / len(CLAIMS):.0%})" for v in VERDICTS],
           textprops=dict(fontsize=FS_BODY))
    ax.text(0, 0, f"{len(CLAIMS)}\nclaims", ha="center", va="center", fontsize=FS_HEAD + 2,
            fontweight="bold")
    ax = fig.add_subplot(gs[0, 1])
    cohorts = [("2022-23", lambda c: c.pub.year <= 2023), ("2024", lambda c: c.pub.year == 2024),
               ("2025", lambda c: c.pub.year == 2025)]
    for i, (lab, sel) in enumerate(cohorts):
        grp = [c for c in CLAIMS if sel(c)]
        left = 0
        for v in VERDICTS:
            n = sum(c.verdict == v for c in grp)
            ax.barh(i, n, left=left, color=V_COLOR[v], height=0.6)
            if n:
                ax.text(left + n / 2, i, str(n), ha="center", va="center", color="white",
                        fontsize=FS_BODY, fontweight="bold")
            left += n
        ax.text(left + 0.3, i, f"{sum(c.verdict == 'expired' for c in grp)} of {len(grp)} expired",
                va="center", fontsize=FS_BODY)
    ax.set_yticks(range(3), [c[0] for c in cohorts], fontsize=FS_BODY)
    ax.set_xlim(0, 28)
    ax.invert_yaxis()
    style(ax, "By publication year", xlabel="claims")
    ax.spines[["top", "right"]].set_visible(False)
    footnote(fig, "Older findings have had longer to age - yet four of the earliest "
             "(C1-C4, 2022-23) are still not expired.")
    return fig


# -------------------------------------------------------
# Step 7: lifetimes
# -------------------------------------------------------
def build_step_7():
    fig, gs = slide("Step 7: Lifetime of each finding", left=0.06, right=0.97, top=0.88,
                    bottom=0.16)
    ax = fig.add_subplot(gs[0, 0])
    order = sorted(CLAIMS, key=lambda c: c.pub)
    for i, c in enumerate(order):
        col = V_COLOR[c.verdict]
        end = c.death or END_OF_PANEL
        ax.plot([to_dt(c.pub), to_dt(end)], [i, i], color=col,
                lw=2.5 if c.death else 1, alpha=1 if c.death else 0.35)
        ax.scatter([to_dt(c.pub)], [i], color=col, s=28, zorder=3)
        if c.death:
            ax.scatter([to_dt(c.death)], [i], marker="|", color=col, s=60, zorder=3)
    ax.axvline(to_dt(END_OF_PANEL), color="0.4", ls="--", lw=1)
    ax.text(to_dt(END_OF_PANEL), len(order) + 0.5, "last model\n2026-09-30", ha="right",
            fontsize=FS_NOTE, va="bottom")
    ax.set_yticks([])
    ax.set_ylim(-1, len(order) + 3)
    ax.invert_yaxis()
    date_axis(ax)
    style(ax, xlabel="dot = publication; thick line = until estimated expiration; "
          "thin = not expired yet")
    ax.spines[["top", "right", "left"]].set_visible(False)
    verdict_legend(ax, loc="lower left")
    lt = lifetimes()
    footnote(fig, f"22 expired claims: mean lifetime {np.mean(lt):.1f} months, median "
             f"{np.median(lt):.1f} (range {min(lt):.0f}-{max(lt):.0f} months).", y=0.05)
    return fig


# -------------------------------------------------------
# Step 8: survival curve
# -------------------------------------------------------
def build_step_8():
    fig, gs = slide("Step 8: How long does a finding last? (Kaplan-Meier, live)", ncols=2,
                    width_ratios=[1.5, 1], wspace=0.1)
    km = kaplan_meier()
    ax = fig.add_subplot(gs[0, 0])
    xs, ys = zip(*km)
    ax.step(list(xs) + [48], [100 * y for y in ys] + [100 * ys[-1]], where="post", color=C_DARK,
            lw=3)
    for k, m in enumerate((6, 12, 24)):
        v = share_expired_by(km, m)
        ax.scatter([m], [100 * v], color=C_ACCENT, zorder=3, s=60)
        ax.annotate(f"{v:.0%} after {m} mo", (m, 100 * v), xytext=(1.5, 63 - 7 * k),
                    fontsize=FS_BODY, fontweight="bold", color=C_ACCENT, va="center",
                    arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=1.2))
    ax.set_xlim(0, 48)
    ax.set_ylim(0, 70)
    style(ax, "Share of claims expired", xlabel="months since publication",
          ylabel="% expired")
    ax.spines[["top", "right"]].set_visible(False)
    ax = fig.add_subplot(gs[0, 1])
    blank(ax)
    text_block(ax, 0.3, 9.6,
               "Why Kaplan-Meier?\n", fs=FS_HEAD, fontweight="bold")
    text_block(ax, 0.3, 8.7,
               "28 claims have not expired yet:\n"
               "we only know they lived at least\n"
               "until Sep 2026 (right-censored).\n"
               "Averaging only the dead ones\n"
               "would be biased.\n\n"
               "Paper: 4% / 16% / 39% expired\n"
               "after 6 / 12 / 24 months.\n\n"
               "Still at full strength (e ≥ 0.8):\n"
               "64% -> 47% -> 31%.",
               fs=FS_BODY)
    footnote(fig, "Durations count from publication, which comes after the experiments - "
             "so they measure how long a published claim stays supported.")
    return fig


def build_step_8_1():
    fig, gs = slide("Step 8.1: Kaplan-Meier in a dozen lines (Part A)", top=0.87, bottom=0.13)
    code_panel(fig.add_subplot(gs[0, 0]), source_of(kaplan_meier), title="kaplan_meier()")
    return fig


# -------------------------------------------------------
# Step 9: where every claim stands
# -------------------------------------------------------
def build_step_9():
    fig, gs = slide("Step 9: How much of each effect remains (fitted e, 2026-09-30)",
                    left=0.07, right=0.98, top=0.88, bottom=0.2)
    ax = fig.add_subplot(gs[0, 0])
    order = sorted(CLAIMS, key=lambda c: c.e_now)
    x = np.arange(len(order))
    ax.axhspan(-1, 0.2, color="#eaeded", zorder=0)
    ax.axhline(1, color="0.4", ls="--", lw=1)
    ax.bar(x, [c.e_now for c in order], color=[V_COLOR[c.verdict] for c in order], width=0.75)
    ax.set_xticks(x, [c.cid for c in order], rotation=90, fontsize=FS_NOTE - 2)
    for cid, txt in [("C11", "fake citations"), ("C13", "reversal curse"),
                     ("C20", "legal halluc."), ("C28", "self-preference"),
                     ("C17", "praise follows user"), ("C49", "prefers AI abstracts\n(reversed)")]:
        i = order.index(CLAIM[cid])
        e = CLAIM[cid].e_now
        ax.annotate(txt, (i, e), xytext=(0, 18 if e >= 0 else -30), textcoords="offset points",
                    ha="center", fontsize=FS_NOTE, arrowprops=dict(arrowstyle="-", color="0.4"))
    ax.set_ylim(-0.9, 2.8)
    style(ax, ylabel="relative effect e")
    ax.spines[["top", "right"]].set_visible(False)
    verdict_legend(ax, loc="upper left")
    footnote(fig, "Weakened is not uniform: prompt injection ~1/4 of the original, reversal curse "
             "~1/2, legal hallucinations ~3/4.", y=0.05)
    return fig


# -------------------------------------------------------
# Step 10-11: what improved and what did not
# -------------------------------------------------------
def build_step_10():
    fig, gs = slide("Step 10: What improved - capability failures fade fast")
    ax = fig.add_subplot(gs[0, 0])
    blank(ax)
    picks = ["C6", "C12", "C36", "C18", "C16", "C7", "C15", "C11", "C35"]
    items = []
    for cid in picks:
        c = CLAIM[cid]
        items.append((f"{c.title}",
                      f"{AREA_SHORT[c.area]}  ·  {c.cite}  ·  expired ~{c.expired_on}  "
                      f"(~{max(1, round(months(c.death, c.pub)))} mo after publication)"))
    row_list(ax, items, top=9.95, step=1.1, face=V_FACE["expired"], edge=V_COLOR["expired"])
    footnote(fig, "Reasoning: 5 expired, 4 weakened, none persist.  These failures have a right "
                  "answer that can be measured and trained against.")
    return fig


def build_step_11():
    fig, gs = slide("Step 11: What did not improve - preferences and judgment")
    ax = fig.add_subplot(gs[0, 0])
    blank(ax)
    groups = [
        ("Models as judges", ["C28", "C31", "C37"]),
        ("Agreeing with / flattering the user", ["C17", "C42", "C43"]),
        ("Sameness", ["C45", "C29"]),
        ("Answering when they should decline", ["C10", "C41"]),
        ("Political lean", ["C21"]),
    ]
    y = 9.8
    for head, cids in groups:
        ax.text(0.2, y, head, fontsize=FS_HEAD, fontweight="bold", color=C_DARK, va="top")
        y -= 0.55
        for cid in cids:
            c = CLAIM[cid]
            ax.text(0.6, y, f"{cid}  {c.title}", fontsize=FS_BODY, va="top", parse_math=False)
            ax.text(11.8, y, f"{c.verdict}  e = {c.e_now:.2f}", fontsize=FS_BODY, va="top",
                    ha="right", color=V_COLOR[c.verdict], fontweight="bold")
            y -= 0.5
        y -= 0.15
    footnote(fig, "Tendencies, not errors on a test with one right answer - harder to notice "
                  "and to train away.  (Social bias is mixed: only C21 persists.)")
    return fig


# -------------------------------------------------------
# Step 12: model-level view
# -------------------------------------------------------
def build_step_12():
    fig, gs = slide("Step 12: Model by model - release date is not the whole story", ncols=2,
                    width_ratios=[1.2, 1], wspace=0.12, left=0.2)
    ax = fig.add_subplot(gs[0, 0])
    names, vals = zip(*MODEL_EXPIRED_COUNTS)
    ax.barh(range(len(vals)), vals, color=["#7f8c8d"] + [C_DARK] * (len(vals) - 1), height=0.6)
    for i, v in enumerate(vals):
        ax.text(v + 0.5, i, f"{v} / 50", va="center", fontsize=FS_BODY, fontweight="bold")
    ax.set_yticks(range(len(vals)), names, fontsize=FS_BODY)
    ax.invert_yaxis()
    ax.set_xlim(0, 36)
    style(ax, "Claims in the expiration zone, per model", xlabel="claims with e ≤ 0.2")
    ax.spines[["top", "right"]].set_visible(False)
    ax = fig.add_subplot(gs[0, 1])
    blank(ax)
    text_block(ax, 0.3, 9.6, "Same quarter (Jul-Sep 2026),\n11 to 30 expired claims.",
               fs=FS_HEAD, fontweight="bold")
    text_block(ax, 0.3, 7.6,
               "Per-model quirks:\n"
               "  Self-preference (C28): Sonnet 5.5\n"
               "    e = 1.09, GPT-6.1 Sol e = 2.88\n"
               "  Longer-answer bias (C27): Sonnet\n"
               "    e = 1.12, Sol e = 0.09\n"
               "  Both still praise what the user\n"
               "    likes (e = 1.85) and pick\n"
               "    non-uniform random numbers.\n"
               "  Fake citations: Qwen3.7-Flash 69%",
               fs=FS_BODY)
    footnote(fig, "A behaviour profile, not a leaderboard.  From GPT-4o mini to GPT-6.1 Sol, "
                  "claims that still hold fall from 30 to 11.")
    return fig


# -------------------------------------------------------
# Step 13: why some aspects improve faster
# -------------------------------------------------------
def build_step_13():
    fig, gs = slide("Step 13: Why some aspects improve faster than others")
    ax = fig.add_subplot(gs[0, 0])
    blank(ax)
    card(ax, 0.2, 3.5, 5.6, 6.4, "Fades fast",
         "Has a right answer\n-> measurable -> trainable\n\n"
         "High-profile failures get\npatched in post-training\n"
         "(strawberry, 'Are you sure?')\n\n"
         "General capability gains:\nreasoning models, long context",
         V_FACE["expired"], V_COLOR["expired"])
    card(ax, 6.2, 3.5, 5.6, 6.4, "Persists or grows",
         "A tendency, not an error:\nno single target to hit\n\n"
         "Some are rewarded by\npreference training\n(agreeing, flattering)\n\n"
         "Judges, raters, sameness,\nfailing to abstain",
         V_FACE["strengthened"], V_COLOR["strengthened"])
    round_box(ax, 0.2, 0.0, 11.6, 3.1, C_MUTE, "0.6", lw=1.2)
    text_block(ax, 0.5, 2.85,
               "In between: structural limits shrink but stay measurable - reversal curse (e = 0.53),\n"
               "two-hop questions (Claude Sonnet 5.5 still misses ~10% vs 43% for ChatGPT), many-turn\n"
               "instructions, prompt injection, invented package names.\n\n"
               "Areas tuned in post-training expire fastest; bias and judgment take longest.",
               fs=FS_BODY)
    footnote(fig, "The 'why' is the authors' hypothesis, not a measured cause; the preference-"
                  "training link is our reading.")
    return fig


# -------------------------------------------------------
# Step 14: implications
# -------------------------------------------------------
def build_step_14():
    fig, gs = slide("Step 14: What this means for anyone citing an LLM finding")
    ax = fig.add_subplot(gs[0, 0])
    blank(ax)
    items = [
        ("A finding is about particular models at a particular time",
         "not a fixed property of 'language models'"),
        ("Report model, version and date with every result",
         "and cite older findings with that caveat"),
        ("Publish prompts and analysis code",
         "so the claim can be re-run on next year's models"),
        ("Re-test before building policy on a claim",
         "e.g. hallucinated references as a detector for AI-written papers"),
        ("Coding agents make re-testing cheap",
         "the authors used Claude Code and Codex: 43 of 48 checked claims reproduced"),
        ("Old is not automatically wrong",
         "C1-C4 (2022-23) still hold in weakened form"),
    ]
    row_list(ax, items, top=9.9, step=1.6)
    return fig


# -------------------------------------------------------
# Step 15: takeaways
# -------------------------------------------------------
def build_step_15():
    fig, gs = slide("Step 15: Takeaways")
    ax = fig.add_subplot(gs[0, 0])
    blank(ax, xlim=(0, 24))
    vc = verdict_counts()
    km = kaplan_meier()
    items = [
        ("44%", f"of 50 claims expired; {vc['weakened']} weakened, {vc['persists']} persist, "
                f"{vc['strengthened']} strengthened."),
        ("16%", f"expired within 1 year, {share_expired_by(km, 24):.0%} within 2 (Kaplan-Meier)."),
        ("~16 mo", "average lifetime of an expired finding."),
        ("Fast", "Reasoning, long context, citations, 'Are you sure?' flips."),
        ("Slow", "Sycophancy, self-preference, sameness, failing to abstain."),
        ("Spread", "Same quarter: 11 to 30 expired claims, by model."),
        ("Practice", "Date your findings; ship the code to re-run them."),
    ]
    for i, (tag, text) in enumerate(items):
        y = 9.4 - i * 1.3
        round_box(ax, 0.2, y - 0.45, 3.6, 0.9, C_DARK, C_DARK, lw=0)
        ax.text(2.0, y, tag, ha="center", va="center", fontsize=FS_BODY, fontweight="bold",
                color="white")
        ax.text(4.4, y, text, ha="left", va="center", fontsize=FS_HEAD + 1, parse_math=False)
    footnote(fig, "What we know about AI has a shelf life.  Check the date on the label.", y=0.07)
    return fig


# -------------------------------------------------------
# Main
# -------------------------------------------------------
BUILDERS = [
    build_preface, build_step_1, build_step_2, build_step_2_1, build_step_3, build_step_3_1,
    build_step_4, build_step_4_1, build_step_4_2, build_step_5, build_step_6, build_step_7, build_step_8,
    build_step_8_1, build_step_9, build_step_10, build_step_11, build_step_12, build_step_13,
    build_step_14, build_step_15,
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The shelf life of LLM findings - lecture deck")
    parser.add_argument("--demo", action="store_true", help="print the analysis in the terminal")
    parser.add_argument("--claims", action="store_true", help="print all 50 claims")
    args = parser.parse_args()
    if args.claims:
        print_claims()
    elif args.demo:
        print_demo()
    else:
        run_slide_deck(BUILDERS)
