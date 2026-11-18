import datetime
import random

import numpy as np
import pandas as pd
import pytest

from cluster_experiments.experiment_analysis import (
    ClusteredOLSAnalysis,
    ExperimentAnalysis,
    GeeExperimentAnalysis,
    RandomizationInferenceConfig,
    RandomizationInferenceResults,
    randomization_p_value,
)
from cluster_experiments.random_splitter import (
    ClusteredSplitter,
    NonClusteredSplitter,
    StratifiedClusteredSplitter,
    SwitchbackSplitter,
)
from cluster_experiments.washover import ConstantWashover


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def clustered_df():
    """Clustered data where treatment is assigned at cluster level."""
    rng = np.random.default_rng(42)
    clusters = [f"C{i}" for i in range(40)]
    rows = []
    for cluster in clusters:
        cluster_effect = rng.normal(0, 1)
        for _ in range(6):
            rows.append(
                {
                    "cluster": cluster,
                    "y": cluster_effect + rng.normal(0, 1),
                }
            )
    df = pd.DataFrame(rows)
    assignment = {c: ("B" if rng.random() < 0.5 else "A") for c in clusters}
    df["treatment"] = df["cluster"].map(assignment)
    return df


@pytest.fixture
def clustered_splitter():
    return ClusteredSplitter(
        cluster_cols=["cluster"], treatment_col="treatment", treatments=["A", "B"]
    )


def clustered_ols(**kwargs):
    """Clustered OLS on the ``cluster``/``y``/``treatment`` fixture schema."""
    return ClusteredOLSAnalysis(cluster_cols=["cluster"], target_col="y", **kwargs)


def _cluster_level_assignments(df):
    """Return per-cluster treatment sets, for verifying cluster-constant labels."""
    return df.groupby("cluster")["treatment"].nunique()


# ---------------------------------------------------------------------------
# The design's re-assignments are legal
# ---------------------------------------------------------------------------
def test_null_reassignments_never_split_a_cluster(
    clustered_df, clustered_splitter
):
    """Re-assignments must respect the cluster: every cluster stays constant."""
    for _ in range(30):
        assigned = clustered_splitter.assign_treatment_df(
            clustered_df.drop(columns=["treatment"])
        )
        assert (_cluster_level_assignments(assigned) == 1).all()


def test_ri_null_distribution_uses_legal_cluster_assignments(
    clustered_df, clustered_splitter
):
    """Every null statistic corresponds to a cluster-constant assignment."""
    # A statistic that returns 1 when a cluster would be split, 0 otherwise.
    def split_indicator(frame):
        return 0.0 if (_cluster_level_assignments(frame) == 1).all() else 1.0

    analysis = ClusteredOLSAnalysis(
        cluster_cols=["cluster"], target_col="y", treatment="B"
    )
    config = RandomizationInferenceConfig(
        splitter=clustered_splitter,
        n_randomizations=25,
        seed=0,
        statistic=split_indicator,
    )

    result = analysis.get_randomization_inference(clustered_df, config)

    assert result.observed_statistic == 0.0
    assert (result.null_statistics == 0.0).all()


def test_stratification_is_respected_in_reassignments():
    """Each null re-assignment stays balanced inside every stratum."""
    rng = np.random.default_rng(0)
    clusters = [f"C{i}" for i in range(40)]
    strata = {c: ("S1" if i % 2 == 0 else "S2") for i, c in enumerate(clusters)}
    df = pd.DataFrame(
        [
            {"cluster": c, "strata": strata[c], "y": rng.normal()}
            for c in clusters
            for _ in range(3)
        ]
    )
    splitter = StratifiedClusteredSplitter(
        cluster_cols=["cluster"],
        strata_cols=["strata"],
        treatment_col="treatment",
        treatments=["A", "B"],
    )

    for _ in range(10):
        assigned = splitter.assign_treatment_df(df)
        unique = assigned.drop_duplicates("cluster")
        counts = unique.groupby(["strata", "treatment"]).size()
        # 20 clusters per stratum, perfectly balanced split
        assert set(counts.values) == {10}


def test_observed_switchback_data_gets_washover_preparation():
    """prepare_observed_df applies truncation + washover without re-randomizing."""
    rng = np.random.default_rng(1)
    hours = pd.date_range("2024-01-01", "2024-01-05 23:00", freq="h")
    raw = pd.DataFrame(
        [
            {"time": t, "city": c, "y": rng.normal()}
            for t in hours
            for c in ["A", "B"]
        ]
    )
    splitter = SwitchbackSplitter(
        time_col="time",
        switch_frequency="1D",
        cluster_cols=["time", "city"],
        washover=ConstantWashover(washover_time_delta=datetime.timedelta(minutes=30)),
    )
    # Build a legal calendar and attach it to the raw hourly rows.
    calendar = splitter.assign_treatment_df(
        raw[["time", "city"]].drop_duplicates(["time", "city"])
    )
    calendar["day"] = calendar["time"].dt.floor("1D")
    mapping = calendar.drop_duplicates(["day", "city"]).set_index(["day", "city"])[
        "treatment"
    ]
    raw["day"] = raw["time"].dt.floor("1D")
    raw["treatment"] = raw.set_index(["day", "city"]).index.map(mapping)

    prepared = splitter.prepare_observed_df(raw)

    # Washover drops rows right after a switch ...
    assert len(prepared) < len(raw)
    # ... keeps the supplied assignment (no random draw) ...
    merged = prepared.merge(
        raw[["time", "city", "treatment"]], on=["time", "city"], how="left"
    )
    assert (merged["treatment_x"] == merged["treatment_y"]).all()
    # ... and is idempotent, so already-split data is accepted.
    assert len(splitter.prepare_observed_df(prepared)) == len(prepared)


def test_switchback_ri_observed_statistic_matches_analysis(clustered_data_switchback):
    df, splitter, analysis = clustered_data_switchback
    result = analysis.get_randomization_inference(
        df,
        RandomizationInferenceConfig(splitter=splitter, n_randomizations=10, seed=3),
    )
    # The observed statistic is exactly the analysis' own point estimate.
    assert result.observed_statistic == pytest.approx(
        analysis.get_point_estimate(df), rel=1e-9
    )
    assert 0 < result.p_value <= 1


def test_switchback_ri_observed_uses_washover_like_null(
    clustered_data_switchback, monkeypatch
):
    """Observed and null statistics are computed on identically prepared data."""
    df, splitter, analysis = clustered_data_switchback
    observed_prepared = []

    def capture(frame, statistic=None):
        observed_prepared.append(frame.copy())
        return 0.0

    monkeypatch.setattr(analysis, "_randomization_statistic", capture)

    analysis.get_randomization_inference(
        df,
        RandomizationInferenceConfig(splitter=splitter, n_randomizations=5, seed=1),
    )

    # The first statistic computed is the observed one; it must already have
    # gone through the switchback truncation + washover preparation.
    observed = observed_prepared[0]
    expected = splitter.prepare_observed_df(df)
    expected = expected[expected["treatment"].isin(["A", "B"])]
    pd.testing.assert_frame_equal(
        observed.reset_index(drop=True).sort_values(["city", "time"]).reset_index(
            drop=True
        ),
        expected.reset_index(drop=True).sort_values(["city", "time"]).reset_index(
            drop=True
        ),
    )


@pytest.fixture
def clustered_data_switchback():
    rng = np.random.default_rng(7)
    hours = pd.date_range("2024-01-01", "2024-01-10 23:00", freq="h")
    raw = pd.DataFrame(
        [
            {"time": t, "city": c, "y": rng.normal()}
            for t in hours
            for c in ["A", "B"]
        ]
    )
    splitter = SwitchbackSplitter(
        time_col="time",
        switch_frequency="1D",
        cluster_cols=["time", "city"],
        washover=ConstantWashover(washover_time_delta=datetime.timedelta(minutes=30)),
    )
    observed = splitter.assign_treatment_df(raw)
    analysis = ClusteredOLSAnalysis(
        cluster_cols=["time", "city"], target_col="y", treatment="B"
    )
    return observed, splitter, analysis


# ---------------------------------------------------------------------------
# p-value rule
# ---------------------------------------------------------------------------
def test_randomization_p_value_rule_two_sided():
    null = np.array([-2.0, -1.0, 0.0, 1.0, 3.0])
    p, n_extreme, n_ties = randomization_p_value(null, 2.0, "two-sided")
    # |null| >= 2 -> [-2.0, 3.0] -> 2 extreme
    assert n_extreme == 2
    assert n_ties == 0
    assert p == pytest.approx(3 / 6)


def test_randomization_p_value_rule_one_sided_directions():
    null = np.array([-2.0, -1.0, 0.0, 1.0, 3.0])
    p_greater, n_greater, _ = randomization_p_value(null, 2.0, "greater")
    p_less, n_less, _ = randomization_p_value(null, 2.0, "less")
    assert n_greater == 1  # only 3.0 is >= 2
    assert n_less == 4  # -2, -1, 0, 1 are <= 2
    assert p_greater == pytest.approx((1 + 1) / 6)
    assert p_less == pytest.approx((1 + 4) / 6)
    # One-sided p-values are not artificially doubled or halved.
    assert p_greater + p_less >= 1.0


def test_randomization_p_value_ties_count_as_extreme():
    null = np.array([1.0, 1.0, 5.0])
    p, n_extreme, n_ties = randomization_p_value(null, 1.0, "two-sided")
    assert n_ties == 2
    assert n_extreme == 3
    assert p == pytest.approx(4 / 4)


def test_randomization_p_value_is_never_zero():
    """Even when no null statistic reaches the observed one, p > 0."""
    null = np.array([0.0, 0.1, 0.2])
    p, n_extreme, _ = randomization_p_value(null, 10.0, "greater")
    assert n_extreme == 0
    assert p == pytest.approx(1 / 4)


def test_ri_result_reports_finite_randomizations_ties_and_rule(
    clustered_df, clustered_splitter
):
    analysis = clustered_ols()
    result = analysis.get_randomization_inference(
        clustered_df,
        RandomizationInferenceConfig(
            splitter=clustered_splitter, n_randomizations=40, seed=11
        ),
    )
    assert isinstance(result, RandomizationInferenceResults)
    assert result.n_randomizations == 40
    assert result.null_statistics.shape == (40,)
    assert result.p_value == pytest.approx(
        (1 + result.n_extreme) / (1 + result.n_randomizations)
    )
    assert "finite randomizations" in result.p_value_rule
    assert result.hypothesis == "two-sided"


def test_ri_one_sided_hypothesis_is_respected(clustered_df, clustered_splitter):
    analysis = clustered_ols()
    result = analysis.get_randomization_inference(
        clustered_df,
        RandomizationInferenceConfig(
            splitter=clustered_splitter,
            n_randomizations=30,
            seed=4,
            hypothesis="greater",
        ),
    )
    assert result.hypothesis == "greater"
    expected_extreme = int((result.null_statistics >= result.observed_statistic).sum())
    assert result.n_extreme == expected_extreme
    assert result.p_value == pytest.approx((1 + result.n_extreme) / 31)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def test_seed_makes_ri_reproducible(clustered_df, clustered_splitter):
    analysis = clustered_ols()
    config = RandomizationInferenceConfig(
        splitter=clustered_splitter, n_randomizations=30, seed=123
    )
    first = analysis.get_randomization_inference(clustered_df, config)
    second = analysis.get_randomization_inference(clustered_df, config)

    np.testing.assert_array_equal(first.null_statistics, second.null_statistics)
    assert first.p_value == second.p_value


def test_ri_does_not_leak_global_random_state(clustered_df, clustered_splitter):
    analysis = clustered_ols()
    random.seed(99)
    np.random.seed(99)
    python_before = random.getstate()
    numpy_before = np.random.get_state()

    analysis.get_randomization_inference(
        clustered_df,
        RandomizationInferenceConfig(
            splitter=clustered_splitter, n_randomizations=10, seed=5
        ),
    )

    assert random.getstate() == python_before
    after = np.random.get_state()
    assert after[0] == numpy_before[0]
    np.testing.assert_array_equal(after[1], numpy_before[1])
    assert after[2] == numpy_before[2]


def test_different_seeds_give_different_null_distributions(
    clustered_df, clustered_splitter
):
    analysis = clustered_ols()
    a = analysis.get_randomization_inference(
        clustered_df,
        RandomizationInferenceConfig(
            splitter=clustered_splitter, n_randomizations=20, seed=1
        ),
    )
    b = analysis.get_randomization_inference(
        clustered_df,
        RandomizationInferenceConfig(
            splitter=clustered_splitter, n_randomizations=20, seed=2
        ),
    )
    assert not np.array_equal(a.null_statistics, b.null_statistics)


# ---------------------------------------------------------------------------
# Same analysis definition on observed and null data
# ---------------------------------------------------------------------------
class CountingOLS(ClusteredOLSAnalysis):
    """OLS that counts how many times the estimator was actually fitted."""

    fit_count = 0

    def analysis_point_estimate(self, df, verbose: bool = False) -> float:
        type(self).fit_count += 1
        return super().analysis_point_estimate(df, verbose=verbose)


def test_ri_refits_analysis_on_every_reassignment(clustered_df, clustered_splitter):
    CountingOLS.fit_count = 0
    analysis = CountingOLS(cluster_cols=["cluster"], target_col="y", treatment="B")
    result = analysis.get_randomization_inference(
        clustered_df,
        RandomizationInferenceConfig(
            splitter=clustered_splitter, n_randomizations=15, seed=2
        ),
    )
    # one fit for the observed statistic + one per valid re-assignment
    assert CountingOLS.fit_count == 1 + result.n_randomizations


def test_non_clustered_splitter_shuffles_units_not_clusters():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"y": rng.normal(size=200)})
    df["treatment"] = np.where(rng.random(200) < 0.5, "B", "A")
    splitter = NonClusteredSplitter(treatment_col="treatment", treatments=["A", "B"])
    analysis = GeeExperimentAnalysis(
        cluster_cols=["cluster"], target_col="y", treatment="B"
    )
    df["cluster"] = np.arange(len(df))
    result = analysis.get_randomization_inference(
        df,
        RandomizationInferenceConfig(splitter=splitter, n_randomizations=20, seed=0),
    )
    assert result.n_randomizations == 20
    assert 0 < result.p_value <= 1


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def test_ri_requires_observed_treatment_column(clustered_splitter):
    df = pd.DataFrame({"cluster": ["a", "b"], "y": [1.0, 2.0]})
    analysis = ClusteredOLSAnalysis(cluster_cols=["cluster"], target_col="y")
    with pytest.raises(ValueError, match="observed assignment"):
        analysis.get_randomization_inference(
            df,
            RandomizationInferenceConfig(
                splitter=clustered_splitter, n_randomizations=5
            ),
        )


def test_ri_validates_treatment_col_alignment(clustered_df):
    splitter = ClusteredSplitter(
        cluster_cols=["cluster"], treatment_col="other_treatment", treatments=["A", "B"]
    )
    analysis = clustered_ols()
    with pytest.raises(ValueError, match="same column the analysis reads"):
        analysis.get_randomization_inference(
            clustered_df,
            RandomizationInferenceConfig(splitter=splitter, n_randomizations=5),
        )


def test_ri_requires_treatment_in_splitter_treatments(
    clustered_df, clustered_splitter
):
    analysis = ClusteredOLSAnalysis(
        cluster_cols=["cluster"], target_col="y", treatment="Z"
    )
    with pytest.raises(ValueError, match="not one of the splitter treatments"):
        analysis.get_randomization_inference(
            clustered_df,
            RandomizationInferenceConfig(
                splitter=clustered_splitter, n_randomizations=5
            ),
        )


def test_ri_requires_control_different_from_treatment(clustered_df, clustered_splitter):
    analysis = clustered_ols()
    with pytest.raises(ValueError, match="control and treatment must be different"):
        analysis.get_randomization_inference(
            clustered_df,
            RandomizationInferenceConfig(
                splitter=clustered_splitter, n_randomizations=5, control="B"
            ),
        )


@pytest.mark.parametrize("n_randomizations", [0, -1, 1.5, "10"])
def test_randomization_config_validates_n_randomizations(
    clustered_splitter, n_randomizations
):
    with pytest.raises(ValueError, match="n_randomizations"):
        RandomizationInferenceConfig(
            splitter=clustered_splitter, n_randomizations=n_randomizations
        )


def test_randomization_config_validates_hypothesis(clustered_splitter):
    with pytest.raises(ValueError):
        RandomizationInferenceConfig(
            splitter=clustered_splitter,
            n_randomizations=5,
            hypothesis="not-a-hypothesis",
        )


def test_randomization_config_validates_statistic(clustered_splitter):
    with pytest.raises(ValueError, match="statistic must be"):
        RandomizationInferenceConfig(
            splitter=clustered_splitter, n_randomizations=5, statistic=42
        )


def test_randomization_config_requires_splitter():
    with pytest.raises(ValueError, match="splitter is required"):
        RandomizationInferenceConfig(splitter=None)


def test_ri_rejects_splitters_that_duplicate_rows(clustered_df):
    from cluster_experiments.random_splitter import RepeatedSampler

    analysis = clustered_ols()
    with pytest.raises(ValueError, match="more rows than it received"):
        analysis.get_randomization_inference(
            clustered_df,
            RandomizationInferenceConfig(
                splitter=RepeatedSampler(treatment_col="treatment"),
                n_randomizations=5,
            ),
        )


def test_ri_rejects_observed_arms_outside_splitter_treatments(
    clustered_df, clustered_splitter
):
    """Observed labels the design cannot generate make the p-value invalid."""
    df = clustered_df.copy()
    df.loc[df.index[:5], "treatment"] = "C"
    analysis = clustered_ols()
    with pytest.raises(ValueError, match="not among the splitter treatments"):
        analysis.get_randomization_inference(
            df,
            RandomizationInferenceConfig(
                splitter=clustered_splitter, n_randomizations=5
            ),
        )


def test_ri_rejects_control_outside_splitter_treatments(
    clustered_df, clustered_splitter
):
    analysis = clustered_ols()
    with pytest.raises(ValueError, match="is not one of the splitter treatments"):
        analysis.get_randomization_inference(
            clustered_df,
            RandomizationInferenceConfig(
                splitter=clustered_splitter, n_randomizations=5, control="Z"
            ),
        )


def test_ri_custom_statistic_is_used(clustered_df, clustered_splitter):
    """A user-supplied statistic replaces the analysis point estimate."""
    analysis = clustered_ols()
    result = analysis.get_randomization_inference(
        clustered_df,
        RandomizationInferenceConfig(
            splitter=clustered_splitter,
            n_randomizations=10,
            seed=0,
            statistic=lambda frame: float(frame["y"].mean()),
        ),
    )
    assert result.statistic_name == "custom"
    # Statistic is a mean of y under whatever assignment, not an ATE contrast
    assert result.observed_statistic == pytest.approx(clustered_df["y"].mean())


def test_ri_custom_statistic_failures_are_discarded_with_warning(
    clustered_df, clustered_splitter
):
    """Undefined re-assignment statistics are skipped, not fatal."""
    calls = {"n": 0}

    def fails_after_observed(frame):
        calls["n"] += 1
        if calls["n"] > 1:  # observed succeeds, every re-assignment fails
            raise ValueError("undefined")
        return 0.0

    analysis = clustered_ols()
    with pytest.raises(ValueError, match="No valid randomization"):
        analysis.get_randomization_inference(
            clustered_df,
            RandomizationInferenceConfig(
                splitter=clustered_splitter,
                n_randomizations=5,
                statistic=fails_after_observed,
            ),
        )


def test_ri_partial_failures_warn(clustered_df, clustered_splitter):
    """A statistic that fails on some re-assignments still returns a result."""
    calls = {"n": 0}

    def fails_sometimes(frame):
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            raise ValueError("undefined")
        return 0.0

    analysis = clustered_ols()
    with pytest.warns(UserWarning, match="discarded"):
        result = analysis.get_randomization_inference(
            clustered_df,
            RandomizationInferenceConfig(
                splitter=clustered_splitter,
                n_randomizations=10,
                seed=0,
                statistic=fails_sometimes,
            ),
        )
    assert result.n_failed_randomizations > 0
    assert result.n_randomizations + result.n_failed_randomizations == 10


def test_ri_summary_mentions_rule(clustered_df, clustered_splitter):
    analysis = clustered_ols()
    result = analysis.get_randomization_inference(
        clustered_df,
        RandomizationInferenceConfig(
            splitter=clustered_splitter, n_randomizations=5, seed=0
        ),
    )
    summary = result.summary()
    assert "Randomization inference results" in summary
    assert "finite randomizations" in summary


def test_ri_statistic_matches_point_estimate_on_observed(
    clustered_df, clustered_splitter
):
    analysis = clustered_ols()
    result = analysis.get_randomization_inference(
        clustered_df,
        RandomizationInferenceConfig(
            splitter=clustered_splitter, n_randomizations=5, seed=0
        ),
    )
    assert result.observed_statistic == pytest.approx(
        analysis.get_point_estimate(clustered_df)
    )
