import numpy as np
import pandas as pd
import pytest

from cluster_experiments.cluster_resampler import (
    ClusterResampler,
    adapt_analysis_for_resampling,
    adapt_splitter_for_resampling,
    resampler_mapping,
)
from cluster_experiments.random_splitter import (
    BalancedClusteredSplitter,
    ClusteredSplitter,
    StratifiedClusteredSplitter,
)


@pytest.fixture
def unit_df():
    # 3 historical units, two within-unit observations each
    return pd.DataFrame(
        {
            "cluster": ["c1", "c1", "c2", "c2", "c3", "c3"],
            "target": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "stratum": ["s1", "s1", "s1", "s1", "s2", "s2"],
        }
    )


@pytest.fixture
def multi_key_df():
    return pd.DataFrame(
        {
            "city": ["A", "A", "A", "B", "B"],
            "day": ["d1", "d1", "d2", "d1", "d2"],
            "target": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


def test_mapping_exposes_cluster_bootstrap():
    assert resampler_mapping["cluster_bootstrap"] is ClusterResampler
    assert resampler_mapping[""] is None


def test_resample_replicates_full_units_and_keeps_distinct_identities(unit_df):
    resampler = ClusterResampler(
        cluster_cols=["cluster"], n_units=6, seed=123
    )
    out = resampler.resample(unit_df, draw_index=0)

    # one draw -> one identity; every draw keeps the full unit (2 rows)
    assert out["resample_id"].nunique() == 6
    assert (out.groupby("resample_id").size() == 2).all()
    # total fan-out: 6 draws x 2 rows
    assert len(out) == 12

    # duplicated draws are NOT merged: resample ids are unique per draw
    assert out["resample_id"].is_unique is False  # 2 rows share each id...
    assert sorted(out.groupby("resample_id").size().unique()) == [2]

    # source identity groups the copies, resample identity separates them
    copied = out.groupby("source_cluster_id")["resample_id"].nunique()
    assert copied.sum() == 6
    assert (copied >= 1).all()


def test_original_columns_and_rows_are_untouched(unit_df):
    resampler = ClusterResampler(cluster_cols=["cluster"], n_units=4, seed=1)
    out = resampler.resample(unit_df, draw_index=0)

    # every original column survives unchanged on each replicated row
    original_cols = ["cluster", "target", "stratum"]
    assert set(original_cols).issubset(out.columns)
    # within a draw the copied observations equal the source rows
    first_draw = out[out["resample_id"] == 0].sort_values("target")
    source = unit_df[unit_df["cluster"] == first_draw["cluster"].iloc[0]].sort_values(
        "target"
    )
    np.testing.assert_array_equal(
        first_draw["target"].to_numpy(), source["target"].to_numpy()
    )

    # row-level provenance points back at source rows
    assert set(out["source_row_id"]).issubset(set(unit_df.index))
    # input df itself was not mutated with identity columns
    assert "resample_id" not in unit_df.columns


def test_resample_is_reproducible_per_draw(unit_df):
    resampler = ClusterResampler(cluster_cols=["cluster"], n_units=6, seed=99)
    first = resampler.resample(unit_df, draw_index=0)
    second = ClusterResampler(
        cluster_cols=["cluster"], n_units=6, seed=99
    ).resample(unit_df, draw_index=0)
    pd.testing.assert_frame_equal(first, second)

    other = ClusterResampler(
        cluster_cols=["cluster"], n_units=6, seed=99
    ).resample(unit_df, draw_index=1)
    assert not first[["source_cluster_id"]].equals(
        other[["source_cluster_id"]]
    )


def test_default_n_units_matches_historical_units(unit_df):
    out = ClusterResampler(cluster_cols=["cluster"], seed=0).resample(
        unit_df, draw_index=0
    )
    assert out["resample_id"].nunique() == 3


def test_stratified_resampling_preserves_stratum_composition(unit_df):
    resampler = ClusterResampler(
        cluster_cols=["cluster"], strata_cols=["stratum"], n_units=6, seed=7
    )
    out = resampler.resample(unit_df, draw_index=0)
    units = out.drop_duplicates("resample_id")
    counts = units.groupby("stratum").size()
    # historical composition: 2 units in s1, 1 in s2 -> doubled
    assert dict(counts) == {"s1": 4, "s2": 2}
    # strata travel with the replicated rows
    assert not out["stratum"].isnull().any()


def test_multi_column_cluster_keys(multi_key_df):
    # 4 unique (city, day) units; one unit has two within-unit observations
    resampler = ClusterResampler(
        cluster_cols=["city", "day"], n_units=20, seed=4
    )
    out = resampler.resample(multi_key_df, draw_index=0)
    assert out["resample_id"].nunique() == 20
    # source identity distinguishes all 4 historical (city, day) units
    assert out["source_cluster_id"].nunique() == 4
    # the original multi-column key is preserved on every copied row
    assert {"city", "day"}.issubset(out.columns)

    # a draw replicates the whole unit: single-row units -> 1 row, the
    # two-row unit -> 2 rows per draw
    rows_per_draw = out.groupby("resample_id").size()
    assert set(rows_per_draw.unique()) == {1, 2}
    two_row_unit = multi_key_df.groupby(["city", "day"]).size()
    two_row_unit = two_row_unit[two_row_unit == 2].index
    drawn_two_row = out.set_index(["city", "day"]).loc[two_row_unit]
    assert (drawn_two_row.groupby("resample_id").size() == 2).all()


def test_missing_cluster_column_raises(unit_df):
    with pytest.raises(ValueError, match="not found"):
        ClusterResampler(cluster_cols=["nope"]).resample(unit_df, draw_index=0)


def test_null_cluster_values_raise():
    df = pd.DataFrame({"cluster": ["a", None], "target": [1.0, 2.0]})
    with pytest.raises(ValueError, match="Null values"):
        ClusterResampler(cluster_cols=["cluster"]).resample(df, draw_index=0)


def test_identity_column_name_clash_raises(unit_df):
    with pytest.raises(ValueError, match="clash"):
        ClusterResampler(
            cluster_cols=["cluster"], resample_id_col="cluster"
        )
    with pytest.raises(ValueError, match="already exist"):
        tagged = unit_df.assign(resample_id=1)
        ClusterResampler(cluster_cols=["cluster"]).resample(
            tagged, draw_index=0
        )


def test_strata_with_multiple_values_per_unit_raises():
    df = pd.DataFrame(
        {"cluster": ["c1", "c1"], "stratum": ["s1", "s2"], "target": [1.0, 2.0]}
    )
    with pytest.raises(ValueError, match="multiple values"):
        ClusterResampler(
            cluster_cols=["cluster"], strata_cols=["stratum"]
        ).resample(df, draw_index=0)


def test_splitter_adaptation_keys_on_resample_identity(unit_df):
    splitter = ClusteredSplitter(cluster_cols=["cluster"])
    resampler = ClusterResampler(cluster_cols=["cluster"], seed=0)
    adapted = adapt_splitter_for_resampling(splitter, resampler)
    assert adapted.cluster_cols == ["resample_id"]
    # original splitter is untouched
    assert splitter.cluster_cols == ["cluster"]

    stratified = StratifiedClusteredSplitter(
        cluster_cols=["cluster"], strata_cols=["stratum"]
    )
    resampler_strat = ClusterResampler(
        cluster_cols=["cluster"], strata_cols=["stratum"], seed=0
    )
    adapted_strat = adapt_splitter_for_resampling(stratified, resampler_strat)
    assert adapted_strat.cluster_cols == ["resample_id"]
    assert adapted_strat.strata_cols == ["stratum"]


def test_analysis_adaptation_keys_on_resample_identity():
    from cluster_experiments.experiment_analysis import ClusteredOLSAnalysis

    analysis = ClusteredOLSAnalysis(cluster_cols=["cluster"])
    resampler = ClusterResampler(cluster_cols=["cluster"], seed=0)
    adapted = adapt_analysis_for_resampling(analysis, resampler)
    assert adapted.cluster_cols == ["resample_id"]
    assert analysis.cluster_cols == ["cluster"]


def test_balanced_splitter_still_balanced_after_adaptation(unit_df):
    resampler = ClusterResampler(
        cluster_cols=["cluster"], n_units=6, seed=3
    )
    splitter = adapt_splitter_for_resampling(
        BalancedClusteredSplitter(cluster_cols=["cluster"]), resampler
    )
    out = resampler.resample(unit_df, draw_index=0)
    split = splitter.assign_treatment_df(out)
    treatments = split.drop_duplicates("resample_id")["treatment"].value_counts()
    assert treatments["A"] == 3 and treatments["B"] == 3
