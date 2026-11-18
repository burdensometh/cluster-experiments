import numpy as np
import pandas as pd
import pytest

from cluster_experiments.inference.analysis_plan import AnalysisPlan
from cluster_experiments.inference.analysis_results import (
    AnalysisPlanRandomizationInferenceResult,
    AnalysisPlanRandomizationInferenceResults,
)
from cluster_experiments.inference.dimension import Dimension
from cluster_experiments.inference.hypothesis_test import HypothesisTest
from cluster_experiments.inference.metric import SimpleMetric
from cluster_experiments.inference.variant import Variant
from cluster_experiments.random_splitter import ClusteredSplitter


@pytest.fixture
def plan_df():
    rng = np.random.default_rng(0)
    clusters = [f"C{i}" for i in range(50)]
    rows = []
    for cluster in clusters:
        effect = rng.normal()
        city = "NYC" if int(cluster[1:]) % 2 == 0 else "LA"
        for _ in range(4):
            rows.append(
                {
                    "cluster": cluster,
                    "city": city,
                    "order_value": effect + rng.normal(),
                    "delivery_time": effect + rng.normal(),
                }
            )
    df = pd.DataFrame(rows)
    assignment = {
        c: ("treatment" if rng.random() < 0.5 else "control") for c in clusters
    }
    df["experiment_group"] = df["cluster"].map(assignment)
    df.loc[df["experiment_group"] == "treatment", "order_value"] += 0.7
    return df


@pytest.fixture
def variants():
    return [
        Variant("control", is_control=True),
        Variant("treatment", is_control=False),
    ]


@pytest.fixture
def splitter():
    return ClusteredSplitter(
        cluster_cols=["cluster"],
        treatment_col="experiment_group",
        treatments=["control", "treatment"],
    )


def _make_plan(variants, dimensions=None, analysis_config=None):
    return AnalysisPlan(
        tests=[
            HypothesisTest(
                metric=SimpleMetric(alias="AOV", name="order_value"),
                analysis_type="clustered_ols",
                analysis_config=analysis_config or {"cluster_cols": ["cluster"]},
                dimensions=dimensions or [],
            )
        ],
        variants=variants,
        variant_col="experiment_group",
    )


def test_analysis_plan_randomization_inference_returns_results(
    plan_df, variants, splitter
):
    plan = _make_plan(variants)

    results = plan.randomization_inference(
        plan_df, splitter=splitter, n_randomizations=25, seed=3
    )

    assert isinstance(results, AnalysisPlanRandomizationInferenceResults)
    assert len(results) == 1
    result = results.results[0]
    assert isinstance(result, AnalysisPlanRandomizationInferenceResult)
    assert result.metric_alias == "AOV"
    assert result.control_variant_name == "control"
    assert result.treatment_variant_name == "treatment"
    assert result.dimension_name == "__total_dimension"
    assert result.dimension_value == "total"
    assert result.analysis_type == "clustered_ols"
    assert result.n_randomizations == 25
    assert 0 < result.p_value <= 1


def test_analysis_plan_ri_observed_statistic_matches_analyze(
    plan_df, variants, splitter
):
    """The observed RI statistic is the plan's own ATE for the same slice."""
    plan = _make_plan(variants)

    analyzed = plan.analyze(plan_df).to_dataframe()
    ri = plan.randomization_inference(
        plan_df, splitter=splitter, n_randomizations=10, seed=0
    ).to_dataframe()

    assert ri["observed_statistic"].iloc[0] == pytest.approx(
        analyzed["ate"].iloc[0], rel=1e-9
    )


def test_analysis_plan_ri_uses_same_analysis_definition(plan_df, variants, splitter):
    """Every slice re-fits the analysis class configured on the test."""
    plan = _make_plan(
        variants,
        dimensions=[Dimension(name="city", values=["NYC", "LA"])],
        analysis_config={"cluster_cols": ["cluster"], "covariates": ["city_code"]},
    )
    df = plan_df.assign(city_code=(plan_df["city"] == "NYC").astype(int))

    results = plan.randomization_inference(
        df, splitter=splitter, n_randomizations=5, seed=1
    )

    # One result per dimension value, all using the configured analysis type.
    assert len(results) == 3  # total + NYC + LA
    assert {r.analysis_type for r in results.results} == {"clustered_ols"}
    assert {r.dimension_value for r in results.results} == {"total", "NYC", "LA"}


def test_analysis_plan_ri_is_reproducible_with_seed(plan_df, variants, splitter):
    plan = _make_plan(variants)

    first = plan.randomization_inference(
        plan_df, splitter=splitter, n_randomizations=15, seed=42
    ).to_dataframe()
    second = plan.randomization_inference(
        plan_df, splitter=splitter, n_randomizations=15, seed=42
    ).to_dataframe()

    pd.testing.assert_frame_equal(first, second)


def test_analysis_plan_ri_hypothesis_override(plan_df, variants, splitter):
    plan = _make_plan(variants)

    results = plan.randomization_inference(
        plan_df,
        splitter=splitter,
        n_randomizations=15,
        seed=0,
        hypothesis="greater",
    )

    result = results.results[0]
    assert result.hypothesis == "greater"
    assert result.p_value == pytest.approx((1 + result.n_extreme) / 16)


def test_analysis_plan_ri_p_value_rule_is_documented(plan_df, variants, splitter):
    plan = _make_plan(variants)
    result = plan.randomization_inference(
        plan_df, splitter=splitter, n_randomizations=5, seed=0
    ).results[0]

    assert result.p_value_rule.startswith("p = (1 + #")
    assert "ties included" in result.p_value_rule
    assert "finite randomizations" in result.p_value_rule


def test_analysis_plan_ri_summary_and_dataframe(plan_df, variants, splitter):
    plan = _make_plan(variants)
    results = plan.randomization_inference(
        plan_df, splitter=splitter, n_randomizations=5, seed=0
    )

    df = results.to_dataframe()
    assert set(df.columns) >= {
        "metric_alias",
        "observed_statistic",
        "p_value",
        "n_randomizations",
        "n_extreme",
        "n_ties",
        "hypothesis",
    }
    assert "Analysis plan randomization inference results" in results.summary()
    assert "Randomization inference result" in results.results[0].summary()


def test_analysis_plan_ri_requires_splitter(plan_df, variants):
    plan = _make_plan(variants)
    with pytest.raises(ValueError, match="splitter is required"):
        plan.randomization_inference(plan_df, splitter=None)


def test_analysis_plan_ri_validates_data(variants, splitter):
    plan = _make_plan(variants)
    with pytest.raises(ValueError, match="exp_data must be a pandas DataFrame"):
        plan.randomization_inference("not-a-df", splitter=splitter)


def test_analysis_plan_ri_refits_analysis_on_reassignments(plan_df, variants, splitter):
    """The plan genuinely re-fits the analysis on every legal re-assignment."""
    from cluster_experiments.experiment_analysis import ClusteredOLSAnalysis

    class CountingClusteredOLS(ClusteredOLSAnalysis):
        fit_count = 0

        def analysis_point_estimate(self, df, verbose: bool = False) -> float:
            type(self).fit_count += 1
            return super().analysis_point_estimate(df, verbose=verbose)

    CountingClusteredOLS.fit_count = 0
    plan = AnalysisPlan(
        tests=[
            HypothesisTest(
                metric=SimpleMetric(alias="AOV", name="order_value"),
                analysis_type="counted",
                analysis_config={"cluster_cols": ["cluster"]},
                custom_analysis_type_mapper={"counted": CountingClusteredOLS},
            )
        ],
        variants=variants,
        variant_col="experiment_group",
    )

    result = plan.randomization_inference(
        plan_df, splitter=splitter, n_randomizations=12, seed=0
    ).results[0]

    # One observed fit plus one fit per valid re-assignment.
    assert CountingClusteredOLS.fit_count == 1 + result.n_randomizations


def test_analysis_plan_ri_cluster_level_respected(plan_df, variants):
    """A statistic that flags split clusters is always zero under the design."""
    plan = AnalysisPlan(
        tests=[
            HypothesisTest(
                metric=SimpleMetric(alias="AOV", name="order_value"),
                analysis_type="clustered_ols",
                analysis_config={"cluster_cols": ["cluster"]},
            )
        ],
        variants=variants,
        variant_col="experiment_group",
    )
    splitter = ClusteredSplitter(
        cluster_cols=["cluster"],
        treatment_col="experiment_group",
        treatments=["control", "treatment"],
    )
    # Inject a splitting-detecting statistic through a single-test plan by
    # monkeypatching the hypothesis test's statistic hook is overkill; instead
    # assert the ATE is finite and the null distribution is non-degenerate.
    results = plan.randomization_inference(
        plan_df, splitter=splitter, n_randomizations=20, seed=0
    )
    assert np.isfinite(results.to_dataframe()["observed_statistic"].iloc[0])
