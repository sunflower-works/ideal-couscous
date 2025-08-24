#!/usr/bin/env python3
"""Final build guard.
Fails (exit 1) if placeholder artefacts, fallback metrics, or missing generated figures/metrics are detected.
Additional checks added:
 - metrics_macros.tex must contain renewcommands for key macros and not just fallback defaults.
 - Accept either .tikz or .tex variants of generated figures.
"""
from __future__ import annotations
import sys
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
FAILURES: list[str] = []

# ---------------- Placeholder scan -----------------
placeholder_pattern = re.compile(r'Placeholder', re.IGNORECASE)
tex_dirs = [ROOT / 'chapters', ROOT / 'figures']
placeholder_hits = []
for d in tex_dirs:
    if not d.exists():
        continue
    for pattern in ('*.tex','*.tikz'):
        for tex in d.rglob(pattern):
            try:
                content = tex.read_text(errors='ignore')
            except Exception:
                continue
            if placeholder_pattern.search(content):
                placeholder_hits.append(tex)
if placeholder_hits:
    FAILURES.append("Found placeholder tokens in: " + ', '.join(str(p.relative_to(ROOT)) for p in placeholder_hits))

# ---------------- Metrics macros existence & content -----------------
metrics_path = ROOT / 'toolset' / 'metrics_macros.tex'
key_patterns = [r'\\renewcommand{\\embedMeanMs}', r'\\renewcommand{\\meanPSNR}', r'\\renewcommand{\\accJPEGtwenty}']
# Fallback defaults copied from macros.tex (if these all appear unchanged treat as fallback)
fallback_values = {
    'embedMeanMs': '148',
    'meanPSNR': '42.6',
    'meanSSIM': '0.984',
    'accJPEGtwenty': '85'
}
if not metrics_path.exists():
    FAILURES.append('Missing toolset/metrics_macros.tex (run make analyze first)')
else:
    content = metrics_path.read_text(errors='ignore')
    if not content.strip():
        FAILURES.append('metrics_macros.tex is empty (analyzer produced no metrics)')
    else:
        for pat in key_patterns:
            if not re.search(pat, content):
                FAILURES.append(f'metrics_macros.tex missing expected macro pattern: {pat}')
                break
        # Extract renewcommand values
        renew_re = re.compile(r'\\renewcommand{\\(\w+)}{([^}]*)}')
        pairs = dict(renew_re.findall(content))
        # Count how many fallback values match exactly
        if pairs:
            fallback_matches = sum(1 for k,v in fallback_values.items() if pairs.get(k) == v)
            # If all fallback keys present and match, treat as failure (likely no real data)
            if fallback_matches == len(fallback_values):
                FAILURES.append('metrics_macros.tex contains only fallback/default values (no updated metrics).')
        else:
            FAILURES.append('metrics_macros.tex has no \renewcommand lines parsed.')

# ---------------- Generated accuracy plot -----------------
fig_dir = ROOT / 'toolset' / 'figures'
accuracy_candidates = list(fig_dir.glob('accuracy_jpeg_generated*.tikz')) + list(fig_dir.glob('accuracy_jpeg_generated*.tex'))
if not accuracy_candidates:
    FAILURES.append('Missing generated accuracy plot (expected toolset/figures/accuracy_jpeg_generated*.tikz).')

# ---------------- Anchoring cost plot (if referenced) ---------------
anchoring_caption_detected = False
for tex in (ROOT / 'chapters').rglob('*.tex'):
    try:
        c = tex.read_text(errors='ignore')
    except Exception:
        continue
    if 'anchoring cost' in c.lower() or 'digest anchoring fee' in c.lower():
        anchoring_caption_detected = True
        break
if anchoring_caption_detected:
    anchoring_candidates = list(fig_dir.glob('anchoring_cost_generated*.tikz')) + list(fig_dir.glob('anchoring_cost_generated*.tex'))
    if not anchoring_candidates:
        FAILURES.append('Anchoring cost referenced but no generated anchoring_cost_generated*.tikz found.')

# ---------------- TODO tokens -----------------
todo_hits = []
for tex in (ROOT / 'chapters').rglob('*.tex'):
    try:
        c = tex.read_text(errors='ignore')
    except Exception:
        continue
    if 'TODO' in c:
        todo_hits.append(tex)
if todo_hits:
    FAILURES.append('Found TODO tokens in: ' + ', '.join(str(p.relative_to(ROOT)) for p in todo_hits))

# ---------------- Result -----------------
if FAILURES:
    print('FINAL BUILD GUARD FAILED:')
    for f in FAILURES:
        print(' -', f)
    sys.exit(1)
print('Final guard passed: no placeholders, updated metrics present, required figures generated.')
