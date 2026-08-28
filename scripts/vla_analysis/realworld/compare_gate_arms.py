#!/usr/bin/env python3
"""Compare two warm-up arms' gate certificates from the RW chain log (E65 overnight ladder).

  python compare_gate_arms.py <chain_log> <ARM_A> <ARM_B>      (B = the newer arm)

Parses, for each arm, the "[gate] expert L<l>: bg X core50 mean Y min-eff Z" and
"[gate] vlm L<l>: min-eff Z bg X" lines that follow "Audit audit_heldout_rw_*_jointwarm_<ARM>_10k
started", prints A vs B per layer with relative change, and a summary verdict on the BINDING
clauses (expert mean core50 <400 layers, expert min-eff <300 layers, VLM min-eff <150 layers):
  IMPROVED  = majority of all metrics up AND mean binding-clause change >= +5% AND no binding
              metric down by more than 5%
  PLATEAU   = majority up but mean binding-clause change < +5%
  REGRESSED = otherwise
Also prints which clauses B still fails. Pure text parsing; safe to run on a copied log.
"""
import re
import sys

log, arm_a, arm_b = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(log, errors="replace").read().replace("\r", "\n")
TH = {"core50": 400.0, "eff": 300.0, "vlm": 150.0}


def parse(arm):
    m = list(re.finditer(rf"^Audit audit_heldout_rw_\S*_jointwarm_{re.escape(arm)}_10k started", text, re.M))
    if not m:
        raise SystemExit(f"no audit section for arm {arm}")
    sec = text[m[-1].start():]
    end = re.search(r"^GATE: (PASS|HARD FAIL)", sec, re.M)
    verdict = end.group(1) if end else "?"
    sec = sec[: end.end()] if end else sec
    out = {}
    for l, bg, c50, eff in re.findall(r"^\[gate\] expert L(\d+): bg ([\d.]+) core50 mean (\d+) min-eff (\d+)", sec, re.M):
        out[f"E{l}"] = {"bg": float(bg), "core50": float(c50), "eff": float(eff)}
    for l, eff, bg in re.findall(r"^\[gate\] vlm L(\d+): min-eff (\d+) bg ([\d.]+)", sec, re.M):
        out[f"V{l}"] = {"vlm": float(eff), "bg": float(bg)}
    return verdict, out


va, A = parse(arm_a)
vb, B = parse(arm_b)
print(f"A = {arm_a} ({va})\nB = {arm_b} ({vb})")
rows, binding, ups, n = [], [], 0, 0
fails = []
for k in sorted(B, key=lambda s: (s[0], int(s[1:]))):
    for met in ("core50", "eff", "vlm"):
        if met not in B[k] or met not in A.get(k, {}):
            continue
        a, b = A[k][met], B[k][met]
        rel = (b - a) / a * 100 if a else float("nan")
        n += 1
        ups += b > a
        is_binding = b < TH[met] or a < TH[met]
        if is_binding:
            binding.append(rel)
        if b < TH[met]:
            fails.append(f"{k}:{met}")
        rows.append(f"  {k:>4} {met:>6}: {a:7.0f} -> {b:7.0f}  {rel:+6.1f}%{'  *' if is_binding else ''}")
    if "bg" in B[k] and "bg" in A.get(k, {}):
        rows.append(f"  {k:>4}     bg: {A[k]['bg']:7.3f} -> {B[k]['bg']:7.3f}")
print("\n".join(rows))
mean_b = sum(binding) / len(binding) if binding else float("nan")
worst_b = min(binding) if binding else float("nan")
if vb == "PASS":
    verdict = "PASS"
elif ups > n / 2 and mean_b >= 5.0 and worst_b >= -5.0:
    verdict = "IMPROVED"
elif ups > n / 2:
    verdict = "PLATEAU"
else:
    verdict = "REGRESSED"
print(f"\nmetrics up {ups}/{n}; binding clauses (*): mean {mean_b:+.1f}%, worst {worst_b:+.1f}%; B still fails: {', '.join(fails) or 'none'}")
print(f"VERDICT: {verdict}")
