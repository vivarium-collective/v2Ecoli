"""Native port of vEcoli ``ecoli/analysis/single/ptools_rna.py``.

Produces a gene × timepoint RNA-count TSV (PathwayTools-compatible format).
Registered in ANALYSIS_REGISTRY as ``"ptools_rna"`` (scale: ``"single"``).

Two v2ecoli parquet shims are applied (see _shims.py):
  - bulk__id + bulk__count  → bulk_count_matrix()  (Shim A)
  - list_sum(n_ribosomes_per_transcript) → active_ribosome  (Shim B)
"""

from __future__ import annotations

import importlib.resources as _ir
import os
from typing import Any

import numpy as np
import pandas as pd
from duckdb import DuckDBPyConnection

from v2ecoli.workflow.analysis import Analysis
from v2ecoli.workflow.analyses._helpers import ptools_heatmap_view
from v2ecoli.workflow.analyses._shims import bulk_count_matrix, ACTIVE_RIBOSOME_SQL


# ---------------------------------------------------------------------------
# Module-level helpers (verbatim from vEcoli, except wd_raw derivation)
# ---------------------------------------------------------------------------

def _flat_dir() -> str:
    """Return the path to the ``reconstruction/ecoli/flat`` directory."""
    import reconstruction.ecoli.flat as _flat_pkg  # installed via vEcoli-sources
    # ir.files() returns a MultiplexedPath; navigate to a file first to get a
    # concrete PosixPath, then take its parent.
    return str((_ir.files(_flat_pkg) / "transcription_units.tsv").parent)


def build_query(columns, history_sql):
    """Generate SQL query for user-specified parquet columns.

    Also carries ``generation`` (a hive partition column) so per-generation
    consolidation can align rows to generation boundaries.
    """
    query_sql = f"""
        SELECT {",".join(columns)}, global_time AS time, generation
        FROM ({history_sql})
        ORDER BY time
    """
    return query_sql


def _groupby_time_keep_generation(outputs_df):
    """Collapse to one row per ``time`` while keeping a per-time ``generation``.

    The data columns are summed (element-wise for list/array columns), matching
    vEcoli semantics. ``generation`` (if present) must NOT be summed — it is a
    label, so it is carried through as the min generation seen at each ``time``
    (one cell per time on a single-daughter lineage, so this is exact). Callers
    that opt into per-generation consolidation read this column back.
    """
    gen_by_time = None
    if "generation" in outputs_df.columns:
        gen_by_time = outputs_df.groupby("time")["generation"].min()
        outputs_df = outputs_df.drop(columns=["generation"])
    outputs_df = outputs_df.groupby("time", as_index=False).sum()
    if gen_by_time is not None:
        outputs_df["generation"] = outputs_df["time"].map(gen_by_time).values
    return outputs_df


def read_outputs(
    history_sql: str,
    conn: DuckDBPyConnection,
    columns=None,
):
    """Retrieve specific columns from parquet outputs and return a DataFrame."""
    if columns is None:
        columns = [
            "bulk__id",
            "bulk__count",
            "listeners__rna_counts__full_mRNA_counts",
        ]
    query_sql = build_query(columns, history_sql)
    outputs_df = conn.sql(query_sql).df()
    # For list/array columns, groupby sum works via element-wise numpy addition.
    # With a single-cell (single scale) query there is typically one row per
    # timestep, so this is effectively identity but preserves vEcoli semantics.
    return _groupby_time_keep_generation(outputs_df)


def retrieve_tu_source(wd_raw):
    """Read and combine transcription units raw data."""
    tu_source_1 = pd.read_csv(
        os.path.join(wd_raw, "transcription_units.tsv"),
        sep="\t",
        header=5,
        index_col=0,
    )
    tu_source_2 = pd.read_csv(
        os.path.join(wd_raw, "transcription_units_added.tsv"),
        sep="\t",
        header=0,
        index_col=0,
    )
    tu_source_2 = tu_source_2.drop("_comments", axis=1)
    tu_source = pd.concat([tu_source_1, tu_source_2], axis=0)
    return tu_source


def tu2gene_mapping(tu_ids, tu_source):
    """Create a mapping between transcription units and individual genes."""
    tu_ids_source = [id_[:-3] for id_ in tu_ids]

    tu_id2genes = []
    for i, tu_id_src in enumerate(tu_ids_source):
        try:
            tu_id_genes = tu_source["genes"][tu_id_src]
        except KeyError:
            tu_id_genes = f"[{tu_id_src.replace('_RNA', '')}]"

        tu_id_genes = tu_id_genes[1:-1].replace('"', "").split(", ")
        tu_id2genes.append(tu_id_genes)

    tu_id2genes = dict(zip(tu_ids, tu_id2genes))
    genes_tu_all = np.unique(
        [gene for genes in list(tu_id2genes.values()) for gene in genes]
    ).tolist()

    return tu_id2genes, genes_tu_all


def get_bulk_ids(sim_data):
    """Return list of bulk molecule IDs in sim_data order."""
    return sim_data.internal_state.bulk_molecules.bulk_data["id"].tolist()


def build_bulk2monomers_matrix(sim_data):
    """Decomplexe bulk species into monomers."""
    bulk_ids = sim_data.internal_state.bulk_molecules.bulk_data["id"].tolist()
    get_monomers = sim_data.process.complexation.get_monomers
    all_monomers = [list(get_monomers(bulk_id)["subunitIds"]) for bulk_id in bulk_ids]
    all_monomers = [item for sublist in all_monomers for item in sublist]
    all_monomers = list(np.unique(all_monomers))

    bulk2monomers = np.zeros((len(bulk_ids), len(all_monomers)))
    for idx, bulk_id in enumerate(bulk_ids):
        monomer_mapping = get_monomers(bulk_id)
        subunits = monomer_mapping["subunitIds"]
        stoich_coeffs = monomer_mapping["subunitStoich"]
        for j in range(len(subunits)):
            subunit = subunits[j]
            monomer_idx = all_monomers.index(subunit)
            bulk2monomers[idx, monomer_idx] = stoich_coeffs[j]

    return bulk2monomers, all_monomers


def consolidate_timepoints(state_mtx, n_tp, normalized=False, generations=None):
    """Consolidate a time-ordered state matrix into fewer columns.

    Default (``generations=None``): ``n_tp`` evenly-spaced checkpoints by tick,
    summing (or averaging, ``normalized=True``) each block between them.

    Per-generation (``generations`` = a per-row generation-label array aligned to
    ``state_mtx`` rows, used by the multigeneration scale): ONE block per
    generation — each output column is the sum/mean over that generation's ticks,
    aligned to generation boundaries (not the drifting evenly-spaced checkpoints).
    Returns ``(blocks, idx)`` where ``idx`` are the first-row indices of each
    generation, so callers' time-column labelling still works.
    """
    if generations is not None:
        gens = np.asarray(generations)
        uniq = sorted(set(gens.tolist()))
        blocks, idx = [], []
        for g in uniq:
            mask = gens == g
            rows = state_mtx[mask]
            block = rows.sum(axis=0) / len(rows) if normalized else rows.sum(axis=0)
            blocks.append(block)
            idx.append(int(np.flatnonzero(mask)[0]))
        return np.stack(blocks, axis=0), np.array(idx, dtype=int)

    checkpoints = np.linspace(0, np.shape(state_mtx)[0] - 1, n_tp, dtype=int)

    if normalized:
        denom = [
            len(state_mtx[checkpoints[i]: checkpoints[i + 1]])
            for i in range(len(checkpoints) - 1)
        ]
        block_sums = [
            state_mtx[checkpoints[i]: checkpoints[i + 1]].sum(axis=0) / denom[i]
            for i in range(len(checkpoints) - 1)
        ]
    else:
        block_sums = [
            state_mtx[checkpoints[i]: checkpoints[i + 1]].sum(axis=0)
            for i in range(len(checkpoints) - 1)
        ]

    block_sums = np.stack(block_sums, axis=0)
    block_sums_final = np.insert(block_sums, 0, state_mtx[0], axis=0)

    return block_sums_final, checkpoints


# ---------------------------------------------------------------------------
# Analysis subclass
# ---------------------------------------------------------------------------

class PtoolsRna(Analysis):
    """Gene × timepoint RNA-count table (PathwayTools-compatible TSV)."""

    name = "ptools_rna"
    scale = "single"
    config_schema = {
        "n_tp": "integer",
        "time_unit": "string",
        "per_generation": "boolean",
        "skip_n_gens": "integer",
    }

    def _do_read_outputs(
        self,
        history_sql: str,
        conn: DuckDBPyConnection,
        columns=None,
    ):
        """Delegate to module-level read_outputs (overridable by mixins)."""
        return read_outputs(history_sql, conn, columns)

    def analyze(
        self,
        *,
        conn: DuckDBPyConnection,
        history_sql: str,
        sim_data,
        variant_metadata: dict[str, Any] | None = None,
        **ctx,
    ) -> dict:
        params = dict(variant_metadata or {})
        params.setdefault("n_tp", 8)
        params.setdefault("time_unit", "minutes")

        if params["time_unit"] not in ("minutes", "seconds"):
            params["time_unit"] = "minutes"

        wd_raw = _flat_dir()

        rna_data = sim_data.process.transcription.rna_data
        tu_source = retrieve_tu_source(wd_raw)
        bulk_ids = get_bulk_ids(sim_data)

        # Shim B: synthesise active_ribosome as sum of per-transcript counts
        output_columns = [
            "bulk__id",
            "bulk__count",
            "listeners__rna_counts__full_mRNA_counts",
            ACTIVE_RIBOSOME_SQL,
        ]

        output_df = self._do_read_outputs(history_sql, conn, output_columns)

        # Shim A: reorder bulk__count columns to sim_data order
        bulk_mtx = bulk_count_matrix(output_df, sim_data)

        # Retrieve mRNAs
        mrna_mtx = np.stack(
            output_df["listeners__rna_counts__full_mRNA_counts"].values
        ).astype(int)

        mrna_tu_ids = rna_data["id"][rna_data["is_mRNA"]].tolist()

        tu2gene_mapping_mrna, genes_tu_mrna = tu2gene_mapping(
            tu_ids=mrna_tu_ids, tu_source=tu_source
        )

        tu_mrna_dict = {}
        for idx, mrna_tu_id in enumerate(mrna_tu_ids):
            tu_mrna_dict[mrna_tu_id] = mrna_mtx[:, idx]

        # Retrieve processed RNAs (tRNAs, rRNAs)
        rna_ids_unprocessed = rna_data["id"][rna_data["is_unprocessed"]]
        rna_ids_mature = sim_data.process.transcription.mature_rna_data["id"]

        # Processed tRNAs
        uncharged_trna_ids = sim_data.process.transcription.uncharged_trna_names
        charged_trna_ids = sim_data.process.transcription.charged_trna_names

        uncharged_trna_bulk_idxs = [bulk_ids.index(i) for i in uncharged_trna_ids]
        charged_trna_bulk_idxs = [bulk_ids.index(i) for i in charged_trna_ids]

        trna_total = (
            bulk_mtx[:, charged_trna_bulk_idxs]
            + bulk_mtx[:, uncharged_trna_bulk_idxs]
        )

        trna_processed_ids = list(
            filter(lambda x: x in rna_ids_mature, uncharged_trna_ids)
        )
        trna_processed_idx = [uncharged_trna_ids.index(i) for i in trna_processed_ids]
        trna_processed_total = trna_total[:, trna_processed_idx]

        rna_processed_total: dict[str, np.ndarray] = {}
        for trna_idx, trna_id in enumerate(trna_processed_ids):
            rna_processed_total[trna_id] = trna_processed_total[:, trna_idx]

        # Add rRNA to rna_processed_total (Shim B: active_ribosome column)
        active_ribosome = output_df["active_ribosome"].values

        processed_rrna_ids = [
            sim_data.molecule_groups.s50_23s_rRNA,
            sim_data.molecule_groups.s30_16s_rRNA,
            sim_data.molecule_groups.s50_5s_rRNA,
        ]
        processed_rrna_ids = [
            item for sublist in processed_rrna_ids for item in sublist
        ]
        processed_rrna_idxs = [bulk_ids.index(i) for i in processed_rrna_ids]

        bulk2monomers, all_monomers = build_bulk2monomers_matrix(sim_data)

        riboprotein_cplxs_ids = ["CPLX0-3953[c]", "CPLX0-3962[c]"]
        riboprotein_cplxs_idxs = [bulk_ids.index(i) for i in riboprotein_cplxs_ids]

        bulk_mtx_riboprotein_cplx = bulk_mtx[:, riboprotein_cplxs_idxs]

        bulk_total_riboprotein_cplx = np.array(
            [
                bulk_mtx_riboprotein_cplx[tp] + active_ribosome[tp]
                for tp in range(len(active_ribosome))
            ]
        )

        bulk_total_riboprotein_monomers = np.matmul(
            bulk_total_riboprotein_cplx, bulk2monomers[riboprotein_cplxs_idxs]
        )

        riboprotein_monomers_idx_rrna = [
            list(all_monomers).index(i) for i in processed_rrna_ids
        ]

        bulk_total_riboprotein_rrna = bulk_total_riboprotein_monomers[
            :, riboprotein_monomers_idx_rrna
        ]

        bulk_total_rrna = bulk_total_riboprotein_rrna + bulk_mtx[:, processed_rrna_idxs]

        for rrna_idx, rrna_id in enumerate(processed_rrna_ids):
            rna_processed_total[rrna_id] = bulk_total_rrna[:, rrna_idx]

        # Reorder processed rRNAs for RNA maturation matrix
        rna_processed: dict[str, np.ndarray] = {}
        for rna_id in rna_ids_mature.tolist():
            rna_processed[rna_id] = rna_processed_total[rna_id]

        rna_processed_mtx = np.stack(list(rna_processed.values())).transpose()

        rna_maturation_stoich_mtx = (
            sim_data.process.transcription.rna_maturation_stoich_matrix.toarray()
        )

        rna_processed_tu = np.matmul(rna_processed_mtx, rna_maturation_stoich_mtx)

        rna_processed_tu_dict: dict[str, np.ndarray] = {}
        for rna_tu_idx, rna_tu in enumerate(rna_ids_unprocessed.tolist()):
            rna_processed_tu_dict[rna_tu] = rna_processed_tu[:, rna_tu_idx]

        rna_processed_tu_ids = list(rna_processed_tu_dict.keys())

        tu2gene_mapping_processed, genes_processed = tu2gene_mapping(
            rna_processed_tu_ids, tu_source
        )

        # Add missing tRNAs
        tu_idx_trna = np.where(rna_data.fullArray()["is_tRNA"])[0]
        tu_idx_not_unprocessed = np.where(~rna_data.fullArray()["is_unprocessed"])[0]
        trna_not_unprocessed_idx = np.intersect1d(tu_idx_trna, tu_idx_not_unprocessed)
        tu_id_trna_missing = rna_data["id"][trna_not_unprocessed_idx].tolist()

        missing_trna_genes = [trna_tu[:4] for trna_tu in tu_id_trna_missing]

        genes_input_raw = pd.read_csv(
            os.path.join(wd_raw, "genes.tsv"), sep="\t", header=5, index_col=0
        )

        missing_trna_gene_ids: dict[str, list[str]] = {}
        missing_trna_genes_biocyc: list[str] = []

        for idx, trna_gene in enumerate(missing_trna_genes):
            gene_id = genes_input_raw.index[
                genes_input_raw["symbol"] == trna_gene
            ][0]
            missing_trna_gene_ids[tu_id_trna_missing[idx]] = [gene_id]
            missing_trna_genes_biocyc.append(gene_id)

        trna_missing_idx = [uncharged_trna_ids.index(i) for i in tu_id_trna_missing]
        trna_missing_counts = trna_total[:, trna_missing_idx]

        trna_missing_tu: dict[str, np.ndarray] = {}
        for idx, trna_id in enumerate(tu_id_trna_missing):
            trna_missing_tu[trna_id] = trna_missing_counts[:, idx]

        # Merge all TU dicts
        tu_dict_full: dict[str, np.ndarray] = {}
        tu_gene_mapping_full: dict[str, list[str]] = {}

        for key in tu_mrna_dict:
            tu_dict_full[key] = tu_mrna_dict[key]
            tu_gene_mapping_full[key] = tu2gene_mapping_mrna[key]

        for key in rna_processed_tu_dict:
            tu_dict_full[key] = rna_processed_tu_dict[key]
            tu_gene_mapping_full[key] = tu2gene_mapping_processed[key]

        for key in trna_missing_tu:
            tu_dict_full[key] = trna_missing_tu[key]
            tu_gene_mapping_full[key] = missing_trna_gene_ids[key]

        tu_genes_all = np.unique(
            genes_tu_mrna + genes_processed + missing_trna_genes_biocyc
        ).tolist()

        tu_gene_mtx = np.zeros([len(tu_dict_full), len(tu_genes_all)])
        for tu_idx, key in enumerate(tu_gene_mapping_full):
            genes_tu = tu_gene_mapping_full[key]
            genes_tu_idx = [tu_genes_all.index(g) for g in genes_tu]
            tu_gene_mtx[tu_idx, genes_tu_idx] = 1

        tu_counts_mtx = np.stack(list(tu_dict_full.values())).transpose()
        rna_counts_gene = np.matmul(tu_counts_mtx, tu_gene_mtx)

        n_tp = int(params["n_tp"])
        gens = (
            output_df["generation"].values
            if params.get("per_generation") and "generation" in output_df.columns
            else None
        )

        rna_counts_gene_blocksum, tp_idx = consolidate_timepoints(
            rna_counts_gene, n_tp, normalized=True, generations=gens
        )

        tp_checkpoints = output_df["time"].values[tp_idx]

        if params["time_unit"] == "minutes":
            tp_checkpoints = tp_checkpoints / 60
            tp_checkpoints = [round(x) for x in tp_checkpoints]

        tp_columns = [str(i) + params["time_unit"][0] for i in tp_checkpoints]

        ptools_rna_df = pd.DataFrame(
            data=rna_counts_gene_blocksum.transpose(),
            columns=tp_columns,
            index=tu_genes_all,
        )
        ptools_rna_df.index.name = "$"

        tsv = ptools_rna_df.to_csv(
            sep="\t", index=True, header=True, float_format="%.4f"
        )
        view = ptools_heatmap_view(ptools_rna_df, "RNA counts (gene × timepoint)")
        return {"data": {"filename": "ptools_rna.tsv", "tsv": tsv}, "view": view}
