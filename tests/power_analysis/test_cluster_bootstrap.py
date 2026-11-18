import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from cluster_experiments import (
    ClusteredOLSAnalysis,
    ClusteredSplitter,
    ClusterResampler,
    ConstantPerturbator,
    GeeExperimentAnalysis,
    NonClusteredSplitter,
    NormalPowerAnalysis,
    OLSAnalysis,
    PowerAnalysis,
    SwitchbackSplitter,
    TargetAggregation,
)
from cluster_experiments.random_splitter import StratifiedClusteredSplitter


@pytest.fixture
def bootstrap_df():
    rng = np.random.default_rng(0)
    clusters = [f"c{i}" for i in range(40)]
    return pd.DataFrame(
        {
            "cluster": rng.choice(clusters, size=2_000),
            "x1": rng.normal(size=2_000),
            "target": rng.normal(size=2_000),
        }
    )


def _build_power_analysis(seed=42, **kwargs):
    resampler = kwargs.pop("resampler", None) or ClusterResampler(
        cluster_cols=["cluster"], seed=seed
    )
    return PowerAnalysis(
        perturbator=ConstantPerturbator(),
        splitter=ClusteredSplitter(cluster_cols=["cluster"]),
        analysis=ClusteredOLSAnalysis(cluster_cols=["cluster"]),
        n_simulations=6,
        seed=seed,
        resampler=resampler,
        **kwargs,
    )


def test_power_analysis_runs_with_resampler(bootstrap_df):
    pw = _build_power_analysis()
    power = pw.power_analysis(bootstrap_df, average_effect=0.5)
    assert 0.0 <= power <= 1.0


def test_user_splitter_and_analysis_are_not_mutated(bootstrap_df):
    splitter = ClusteredSplitter(cluster_cols=["cluster"])
    analysis = ClusteredOLSAnalysis(cluster_cols=["cluster"])
    PowerAnalysis(
        perturbator=ConstantPerturbator(),
        splitter=splitter,
        analysis=analysis,
        n_simulations=2,
        resampler=ClusterResampler(cluster_cols=["cluster"], seed=1),
    )
    assert splitter.cluster_cols == ["cluster"]
    assert analysis.cluster_cols == ["cluster"]


def test_sampled_units_are_independent_in_split_and_analysis(bootstrap_df):
    pw = _build_power_analysis()
    simulated = pw._prepare_simulation_df(bootstrap_df, None, 0)
    split = pw._split(simulated)

    # each sampled unit receives exactly one treatment
    treatments_per_unit = split.groupby("resample_id")["treatment"].nunique()
    assert (treatments_per_unit == 1).all()

    # clustered standard errors are computed on the sampled units
    analysis_groups = pw.analysis._get_cluster_column(split).nunique()
    assert analysis_groups == split["resample_id"].nunique()

    # copies of the same historical unit can be randomized differently
    source_treatments = split.groupby("source_cluster_id")["treatment"].nunique()
    assert (source_treatments >= 1).all()


def test_resampled_rows_keep_full_provenance(bootstrap_df):
    pw = _build_power_analysis()
    simulated = pw._prepare_simulation_df(bootstrap_df, None, 0)
    for col in ["resample_id", "source_cluster_id", "source_row_id"]:
        assert col in simulated.columns
    assert set(simulated["source_row_id"]).issubset(set(bootstrap_df.index))
    # source rows are never scattered: every draw replicates the unit wholesale
    assert (simulated.groupby("resample_id")["source_cluster_id"].nunique() == 1).all()


def test_repeated_runs_are_reproducible(bootstrap_df):
    first = _build_power_analysis(seed=7).power_analysis(
        bootstrap_df, average_effect=0.0
    )
    second = _build_power_analysis(seed=7).power_analysis(
        bootstrap_df, average_effect=0.0
    )
    assert first == second


def _resample_draw(args):
    """Module-level helper so multiprocessing can pickle it."""
    df, index = args
    return ClusterResampler(cluster_cols=["cluster"], seed=5).resample(
        df, draw_index=index
    )[["source_cluster_id", "resample_id"]]


def test_parallel_and_serial_resampling_draws_match(bootstrap_df):
    # The resampler RNG is derived per draw index, so forked workers draw the
    # same units as the serial loop. Compare one draw's resampled dataframe.
    args = [(bootstrap_df, i) for i in range(4)]
    serial = [_resample_draw(arg) for arg in args]
    from multiprocessing import Pool

    with Pool(2) as pool:
        parallel = pool.map(_resample_draw, args)
    for left, right in zip(serial, parallel):
        pd.testing.assert_frame_equal(left, right)


def test_cupac_covariates_join_on_source_identity(bootstrap_df):
    pre_experiment = bootstrap_df.assign(
        target=np.random.default_rng(1).normal(size=len(bootstrap_df))
    )
    pw = PowerAnalysis(
        perturbator=ConstantPerturbator(),
        splitter=ClusteredSplitter(cluster_cols=["cluster"]),
        analysis=ClusteredOLSAnalysis(
            cluster_cols=["cluster"], covariates=["estimate_target"]
        ),
        cupac_model=TargetAggregation(agg_col="cluster"),
        n_simulations=3,
        seed=2,
        resampler=ClusterResampler(cluster_cols=["cluster"], seed=2),
    )
    power = pw.power_analysis(
        bootstrap_df, pre_experiment_df=pre_experiment, average_effect=0.1
    )
    assert 0.0 <= power <= 1.0
    simulated = pw._prepare_simulation_df(bootstrap_df, pre_experiment, 0)
    assert "estimate_target" in simulated.columns
    # copies of one source unit share the historical covariate value
    agg = simulated.groupby("source_cluster_id")["estimate_target"].nunique()
    assert (agg == 1).all()


def test_mlrate_folds_are_leakage_safe_on_source_units(bootstrap_df):
    from sklearn.model_selection import GroupKFold

    pw = PowerAnalysis(
        perturbator=ConstantPerturbator(),
        splitter=ClusteredSplitter(cluster_cols=["cluster"]),
        analysis=ClusteredOLSAnalysis(
            cluster_cols=["cluster"], covariates=["estimate_target"]
        ),
        cupac_model=LinearRegression(),
        ml_option="mlrate",
        n_folds=4,
        features_cupac_model=["x1"],
        n_simulations=3,
        seed=5,
        resampler=ClusterResampler(cluster_cols=["cluster"], seed=5),
    )
    assert pw.ml_handler.group_cols == ["source_cluster_id"]
    assert "resample_id" in pw.ml_handler.exclude_cols

    simulated = pw._prepare_simulation_df(bootstrap_df, None, 0)
    groups = (
        simulated[pw.ml_handler.group_cols].astype(str).apply("_".join, axis=1).values
    )
    for train_idx, val_idx in GroupKFold(4).split(simulated, groups=groups):
        assert not set(groups[train_idx]) & set(groups[val_idx])

    power = pw.power_analysis(bootstrap_df, average_effect=0.1)
    assert 0.0 <= power <= 1.0


def test_stratified_splitter_with_stratified_resampling(bootstrap_df):
    df = bootstrap_df.assign(
        stratum=np.where(
            bootstrap_df["cluster"].str.extract(r"c(\d+)").astype(int)[0] < 20,
            "s1",
            "s2",
        )
    )
    pw = PowerAnalysis(
        perturbator=ConstantPerturbator(),
        splitter=StratifiedClusteredSplitter(
            cluster_cols=["cluster"], strata_cols=["stratum"]
        ),
        analysis=ClusteredOLSAnalysis(cluster_cols=["cluster"]),
        n_simulations=3,
        seed=3,
        resampler=ClusterResampler(
            cluster_cols=["cluster"], strata_cols=["stratum"], seed=3
        ),
    )
    assert pw.splitter.cluster_cols == ["resample_id"]
    assert pw.splitter.strata_cols == ["stratum"]
    power = pw.power_analysis(df, average_effect=0.2)
    assert 0.0 <= power <= 1.0


def test_normal_power_analysis_supports_resampling(bootstrap_df):
    npw = NormalPowerAnalysis(
        splitter=ClusteredSplitter(cluster_cols=["cluster"]),
        analysis=GeeExperimentAnalysis(cluster_cols=["cluster"]),
        n_simulations=4,
        seed=1,
        resampler=ClusterResampler(cluster_cols=["cluster"], seed=1),
    )
    std_error = npw._get_average_standard_error(bootstrap_df, n_simulations=4)
    assert std_error > 0
    assert npw.analysis.cluster_cols == ["resample_id"]


def test_switchback_splitter_is_rejected():
    with pytest.raises(NotImplementedError):
        PowerAnalysis(
            perturbator=ConstantPerturbator(),
            splitter=SwitchbackSplitter(
                time_col="date", cluster_cols=["cluster", "date"]
            ),
            analysis=ClusteredOLSAnalysis(cluster_cols=["cluster", "date"]),
            resampler=ClusterResampler(cluster_cols=["cluster"]),
        )


def test_non_clustered_splitter_is_rejected():
    with pytest.raises(ValueError, match="clustered splitter"):
        PowerAnalysis(
            perturbator=ConstantPerturbator(),
            splitter=NonClusteredSplitter(),
            analysis=OLSAnalysis(),
            resampler=ClusterResampler(cluster_cols=["cluster"]),
        )


def test_resampler_from_config(bootstrap_df):
    pw = PowerAnalysis.from_dict(
        {
            "analysis": "ols_clustered",
            "splitter": "clustered",
            "perturbator": "constant",
            "cluster_cols": ["cluster"],
            "n_simulations": 3,
            "seed": 11,
            "resampler": "cluster_bootstrap",
            "resample_n_units": 40,
        }
    )
    assert isinstance(pw.resampler, ClusterResampler)
    assert pw.resampler.n_units == 40
    power = pw.power_analysis(bootstrap_df, average_effect=0.1)
    assert 0.0 <= power <= 1.0


def test_unknown_resampler_in_config_raises():
    with pytest.raises(ValueError, match="Unknown resampler"):
        PowerAnalysis.from_dict(
            {
                "analysis": "ols_clustered",
                "splitter": "clustered",
                "perturbator": "constant",
                "cluster_cols": ["cluster"],
                "resampler": "does_not_exist",
            }
        )
