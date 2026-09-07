"""Render the EcoCyc <-> model coverage comparison as a standalone HTML figure.

Reads the numbers from promoter-specific-regulation's recorded
ecocyc_model_coverage block, so the figure cannot drift from the study record.
Regenerate after re-running analyses/ecocyc_model_coverage.py.
"""
from __future__ import annotations
import html as H
import json
import pathlib
import yaml

STUDY = pathlib.Path("workspace/studies/promoter-specific-regulation/study.yaml")
OUT = pathlib.Path("reports/figures/promoter-specific-regulation/ecocyc_model_coverage.html")

cov = yaml.safe_load(STUDY.read_text())["ecocyc_model_coverage"]
tu, tf = cov["transcription_units"], cov["transcription_factors"]

ROWS = [
    ("Transcription units", None, None, None, True),
    ("In EcoCyc 30.0 (transunits.dat)", tu["ecocyc_transunits_dat"], None, None, False),
    ("In the model's input file", tu["model_input_tsv"],
     f"{tu['source_tus_present_in_ecocyc']}/{tu['model_input_tsv']} exist in EcoCyc",
     tu["source_tus_present_in_ecocyc"] / tu["model_input_tsv"], False),
    ("In the model itself", tu["model_rna_data"],
     f"{tu['absent_from_model_total']} absent, all lost inside the model build",
     tu["model_rna_data"] / tu["model_input_tsv"], False),
    ("  — removed as unsupported types", tu["explicitly_removed"], None, None, False),
    ("  — not kept thereafter", tu["not_kept_thereafter"],
     "dominated by the gene-set dedup", None, False),
    ("  — dedup casualties (all source TUs)", tu["dedup_casualties_all_source"],
     "624 excluding the 64 removed; 13 TUs in both sets", None, False),
    ("  — single-gene TUs, relabelled not lost", tu["single_gene_renamed"], None, None, False),
    ("Transcription factors", None, None, None, True),
    ("Distinct regulators in EcoCyc", tf["ecocyc_distinct_regulators"], None, None, False),
    ("Transcription factors in the model", tf["model_tfs"],
     "a curated subset: the TFs with condition data ParCa can fit",
     tf["model_tfs"] / tf["ecocyc_distinct_regulators"], False),
    ("Model TFs resolvable in EcoCyc", tf["model_tfs_found_in_ecocyc"],
     f"{tf['model_tfs_found_without_aliases']}/23 without the three id aliases", 1.0, False),
    ("EcoCyc TF-binding records", tf["ecocyc_tf_binding_records"], None, None, False),
    ("  — concerning the model's TFs", tf["records_belonging_to_model_tfs"], None,
     tf["records_belonging_fraction"], False),
    ("Model's declared (TU, TF) edges", tf["model_declared_edges"],
     "the only things the promoter-specific correction can touch", None, False),
]

def bar(frac):
    if frac is None:
        return ""
    pct = frac * 100
    hue = "#2563eb" if frac > 0.5 else "#b45309"
    return (f'<div class="bw"><div class="bf" style="width:{pct:.1f}%;background:{hue}"></div>'
            f'</div><span class="pc">{pct:.1f}%</span>')

body = []
for label, n, note, frac, is_head in ROWS:
    if is_head:
        body.append(f'<tr class="head"><td colspan="4">{H.escape(label)}</td></tr>')
        continue
    indent = ' class="ind"' if label.startswith("  ") else ""
    body.append(
        f'<tr><td{indent}>{H.escape(label.strip())}</td>'
        f'<td class="num">{n:,}</td>'
        f'<td class="frac">{bar(frac)}</td>'
        f'<td class="note">{H.escape(note or "")}</td></tr>')

OUT.write_text(f"""<!doctype html><meta charset="utf-8">
<title>EcoCyc vs model coverage</title>
<style>
 body{{font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:18px;color:#111}}
 h1{{font-size:17px;margin:0 0 2px}} p.sub{{margin:0 0 16px;color:#555;font-size:13px;max-width:70em}}
 table{{border-collapse:collapse;width:100%;max-width:70em}}
 td{{padding:6px 10px;border-bottom:1px solid #eee;vertical-align:middle}}
 tr.head td{{background:#f4f6f8;font-weight:600;border-bottom:1px solid #d8dde2;padding-top:10px}}
 td.ind{{padding-left:26px;color:#555}}
 td.num{{text-align:right;font-variant-numeric:tabular-nums;font-weight:600;width:7em}}
 td.frac{{width:14em;white-space:nowrap}} td.note{{color:#555;font-size:12.5px}}
 .bw{{display:inline-block;width:9em;height:9px;background:#e8ebee;border-radius:5px;overflow:hidden;
      vertical-align:middle}}
 .bf{{height:100%}} .pc{{margin-left:7px;font-size:12px;color:#444;font-variant-numeric:tabular-nums}}
 .foot{{margin-top:14px;color:#555;font-size:12.5px;max-width:70em}}
 code{{background:#f4f6f8;padding:1px 4px;border-radius:3px;font-size:12px}}
</style>
<h1>EcoCyc 30.0 vs the model — coverage in both directions</h1>
<p class="sub">The two are incomplete in <b>opposite</b> directions. The model holds almost every
EcoCyc transcription unit but only a thin slice of its regulators, and that asymmetry is what
determines which regulatory edges a positional correction may legitimately change.</p>
<table>{''.join(body)}</table>
<p class="foot"><b>Why it matters.</b> With the model carrying
{tf['model_tfs']}/{tf['ecocyc_distinct_regulators']} of EcoCyc's regulators, and EcoCyc itself
incomplete for any given promoter, &ldquo;no record here&rdquo; is weak evidence — which is why the
promoter-specific rule deletes only edges EcoCyc actively <i>contradicts</i> and leaves unrecorded
ones alone. Deleting on absence is what made the original blind rule strip 89.5% of the network.
<br>Measured {cov['measured']} by <code>{cov['script']}</code>; figure regenerated by
<code>scripts/build_ecocyc_coverage_figure.py</code>.</p>
""")
print(f"wrote {OUT}")
