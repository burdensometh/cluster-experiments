"""
Cluster-level resampling for power simulations.

Power analyses built from historical data need to re-sample the historical
clusters *with replacement* to approximate the sampling distribution of the
experiment. A cluster drawn twice must remain two independent sampling units:
its within-unit observations are copied wholesale and the copies receive
distinct resampling identities, so downstream splitters and clustered standard
errors never merge the copies in a ``groupby``.

The original cluster identity is always kept alongside the resampling
identity. It is what ties the copied rows back to their source unit, e.g. to
join historical covariates or to define the leakage-safe cross-fitting groups
for variance reducers.
"""

from copy import copy
from typing import List, Optional

import numpy as np
import pandas as pd


class ClusterResampler:
    """
    Resamples the units defined by ``cluster_cols`` with replacement and
    fully copies every within-unit observation for each draw.

    The resampled dataframe keeps every original column untouched and gains
    identity columns:

    - ``resample_id_col``: unique identity of the *sampled* unit. Two draws of
      the same historical cluster get different values here, so they behave as
      two independent units in the splitter, the perturbator and the clustered
      standard errors.
    - ``source_id_col``: identity of the historical unit the draw was taken
      from. Two draws of the same historical cluster share this value, which is
      what allows historical covariates to be joined and leakage-safe folds to
      be defined at the source-unit level.
    - ``source_index_col``: the original row index (row-level provenance, so
      every simulated row can be traced back to a source row).

    Strata columns travel with the unit: every copied row carries the strata
    values of its source unit. When ``strata_cols`` is given, units are drawn
    with replacement *within each stratum*, preserving the historical number of
    units per stratum.

    Arguments:
        cluster_cols: columns defining the historical sampling units.
        n_units: number of units to draw. If None, the historical number of
            unique units is used.
        strata_cols: optional columns defining resampling strata.
        resample_id_col: name of the column holding the sampled-unit identity.
        source_id_col: name of the column holding the source-unit identity.
        source_index_col: name of the column holding the source row index.
        seed: optional seed for reproducible resampling.

    Usage:
    ```python
    import pandas as pd
    from cluster_experiments.cluster_resampler import ClusterResampler

    df = pd.DataFrame(
        {
            "cluster": ["c1", "c1", "c2", "c2"],
            "target": [1.0, 2.0, 3.0, 4.0],
        }
    )

    resampler = ClusterResampler(cluster_cols=["cluster"], seed=123)
    resampled_df = resampler.resample(df)
    ```
    """

    def __init__(
        self,
        cluster_cols: List[str],
        n_units: Optional[int] = None,
        strata_cols: Optional[List[str]] = None,
        resample_id_col: str = "resample_id",
        source_id_col: str = "source_cluster_id",
        source_index_col: str = "source_row_id",
        seed: Optional[int] = None,
    ) -> None:
        if not cluster_cols:
            raise ValueError(
                "ClusterResampler requires at least one column in cluster_cols."
            )
        self.cluster_cols = cluster_cols
        self.n_units = n_units
        self.strata_cols = strata_cols or []
        self.resample_id_col = resample_id_col
        self.source_id_col = source_id_col
        self.source_index_col = source_index_col
        self.seed = seed
        self.identity_cols = [
            self.resample_id_col,
            self.source_id_col,
            self.source_index_col,
        ]
        self._rng: Optional[np.random.Generator] = None
        self._check_column_names()

    @classmethod
    def from_config(cls, config) -> "ClusterResampler":
        """Creates a ClusterResampler from a PowerConfig."""
        return cls(
            cluster_cols=config.cluster_cols,
            n_units=config.resample_n_units,
            strata_cols=config.strata_cols,
            seed=config.seed,
        )

    def resample(
        self, df: pd.DataFrame, draw_index: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Returns a resampled copy of ``df`` where units defined by
        ``cluster_cols`` are drawn with replacement and all of their rows are
        replicated for every draw.

        Arguments:
            df: dataframe with the historical observations.
            draw_index: optional index of the simulation draw. When given, the
                RNG is derived deterministically from ``(seed, draw_index)``,
                which keeps parallel simulations independent yet reproducible.
        """
        self._check_input_columns(df)
        df = df.copy()

        # Row-level provenance: keep the original index before any fan-out.
        df[self.source_index_col] = df.index.to_numpy()

        units = self._get_units(df)
        draws = self._draw_units(units, self._get_rng(draw_index))
        draws[self.resample_id_col] = np.arange(len(draws))

        # Attach the source-unit identity at row level. This is a many-to-one
        # lookup on the cluster key, so rows are not duplicated here.
        unit_lookup = units[self.cluster_cols + [self.source_id_col]].drop_duplicates()
        df = df.merge(unit_lookup, on=self.cluster_cols, how="left")

        # Merge the draws on the cluster key only: a unit drawn k times fans
        # its rows out into k full, independent copies, each with its own
        # resample identity. The source identity and strata travel with every
        # copy on the source rows, so they are not merged again here.
        draw_lookup = draws[self.cluster_cols + [self.resample_id_col]]
        resampled = df.merge(draw_lookup, on=self.cluster_cols, how="inner")
        return resampled.reset_index(drop=True)

    def _get_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """Unique historical units with their strata attached."""
        units = df.loc[:, self.cluster_cols + self.strata_cols].drop_duplicates()
        units = units.reset_index(drop=True)
        units[self.source_id_col] = np.arange(len(units))

        if self.strata_cols:
            # A unit must belong to a single stratum, otherwise the strata
            # cannot be carried along with the replicated rows.
            for strata_col in self.strata_cols:
                n_strata_per_unit = units.groupby(self.cluster_cols)[
                    strata_col
                ].nunique()
                if n_strata_per_unit.max() > 1:
                    raise ValueError(
                        f"There are multiple values in {strata_col} for the same "
                        "cluster item. You cannot stratify resampling on this column."
                    )
        return units

    def _draw_units(
        self, units: pd.DataFrame, rng: np.random.Generator
    ) -> pd.DataFrame:
        """Draws units with replacement, within strata if provided."""
        n_units = self.n_units or len(units)

        if not self.strata_cols:
            draw_positions = rng.integers(0, len(units), size=n_units)
            return units.iloc[draw_positions].reset_index(drop=True)

        # Stratified bootstrap: resample within each stratum so that the number
        # of units per stratum matches the historical composition.
        n_per_stratum = units.groupby(self.strata_cols, as_index=False).size()
        if n_units != len(units):
            if n_units % len(n_per_stratum) != 0:
                raise ValueError(
                    "When resampling with strata_cols, n_units must be a multiple "
                    f"of the number of strata ({len(n_per_stratum)}), got {n_units}."
                )
            multiplier = n_units // len(units)
            n_per_stratum["size"] = n_per_stratum["size"] * multiplier

        drawn = []
        for _, stratum_row in n_per_stratum.iterrows():
            mask = np.ones(len(units), dtype=bool)
            for strata_col in self.strata_cols:
                mask &= units[strata_col].eq(stratum_row[strata_col]).to_numpy()
            stratum_units = units.loc[mask]
            draw_positions = rng.integers(
                0, len(stratum_units), size=int(stratum_row["size"])
            )
            drawn.append(stratum_units.iloc[draw_positions])
        return pd.concat(drawn).reset_index(drop=True)

    def _get_rng(
        self, draw_index: Optional[int] = None
    ) -> np.random.Generator:
        """
        RNG for a resampling draw. With an explicit draw index the stream is
        derived from ``(seed, draw_index)`` so forked worker processes running
        different draws stay independent while results stay reproducible.
        Without an index a process-local persistent stream is returned.
        """
        if draw_index is not None and self.seed is not None:
            return np.random.default_rng((self.seed, draw_index))
        if self._rng is None:
            self._rng = np.random.default_rng(self.seed)
        return self._rng

    def _check_column_names(self) -> None:
        reserved = set(self.identity_cols)
        data_cols = set(self.cluster_cols + self.strata_cols)
        overlap = reserved.intersection(data_cols)
        if overlap:
            raise ValueError(
                f"Resampler identity columns {sorted(overlap)} clash with input "
                "cluster/strata columns. Rename the identity column(s)."
            )
        if len(self.identity_cols) != len(reserved):
            raise ValueError(
                "resample_id_col, source_id_col and source_index_col must have "
                "distinct names."
            )

    def _check_input_columns(self, df: pd.DataFrame) -> None:
        missing = [
            col
            for col in self.cluster_cols + self.strata_cols
            if col not in df.columns
        ]
        if missing:
            raise ValueError(
                f"ClusterResampler: columns {missing} not found in input dataframe."
            )
        present_identity = [col for col in self.identity_cols if col in df.columns]
        if present_identity:
            raise ValueError(
                f"ClusterResampler identity columns {present_identity} already exist "
                "in the input dataframe. Rename the resampler identity column(s)."
            )
        if df[self.cluster_cols].isnull().values.any():
            raise ValueError(
                f"Null values found in cluster_cols: {self.cluster_cols}. "
                "Please remove nulls before resampling."
            )

    def __repr__(self) -> str:
        return (
            f"ClusterResampler(cluster_cols={self.cluster_cols!r}, "
            f"strata_cols={self.strata_cols!r}, n_units={self.n_units}, "
            f"seed={self.seed})"
        )


resampler_mapping = {
    "": None,
    "cluster_bootstrap": ClusterResampler,
}


def adapt_splitter_for_resampling(
    splitter, resampler: ClusterResampler
):
    """
    Returns a splitter copy whose randomization unit is the sampled-unit
    identity (``resample_id_col``) instead of the historical cluster columns.

    The original splitter is never mutated. For stratified splitters, the
    strata columns are preserved, so balancing happens across the sampled units
    while strata information travels with the replicated rows.
    """
    adapted = copy(splitter)
    adapted.cluster_cols = [resampler.resample_id_col]
    if hasattr(adapted, "strata_cols") and resampler.strata_cols:
        adapted.strata_cols = resampler.strata_cols
    return adapted


def adapt_analysis_for_resampling(
    analysis, resampler: ClusterResampler
):
    """
    Returns an analysis copy whose clustering (and therefore clustered standard
    errors) is computed on the independent sampled units, not on the source
    units they were drawn from.
    """
    adapted = copy(analysis)
    adapted.cluster_cols = [resampler.resample_id_col]
    return adapted
