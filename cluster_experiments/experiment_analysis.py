import logging
import random
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pandas.api.types import is_numeric_dtype
from scipy.stats import norm, ttest_ind, ttest_rel

from cluster_experiments.relative_lift_transformer import (
    LiftRegressionTransformer,
    RegressionResultsProtocol,
)
from cluster_experiments.synthetic_control_utils import get_w
from cluster_experiments.utils import (
    HypothesisEntries,
    ModelResults,
    _original_time_column,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from cluster_experiments.random_splitter import RandomSplitter


@dataclass
class ConfidenceInterval:
    """
    Class to define the structure of a confidence interval.
    """

    lower: float
    upper: float
    alpha: float

    def __str__(self) -> str:
        """
        Usage:
        ```python
        from cluster_experiments import ConfidenceInterval
        ci = ConfidenceInterval(lower=0.1, upper=0.5, alpha=0.05)
        print(ci)
        ```
        """
        return f"[{self.lower:.4f}, {self.upper:.4f}] (alpha={self.alpha})"

    def summary(self) -> str:
        """Return a short summary of the confidence interval.

        Usage:
        ```python
        from cluster_experiments import ConfidenceInterval
        ci = ConfidenceInterval(lower=0.1, upper=0.5, alpha=0.05)
        print(ci.summary())
        ```
        """
        return (
            f"Confidence interval (1 - alpha = {1 - self.alpha:.2%})\n"
            f"  Lower: {self.lower:.6g}\n"
            f"  Upper: {self.upper:.6g}"
        )


@dataclass
class InferenceResults:
    """
    Class to define the structure of complete statistical analysis results.

    When the analysis uses a fitted model (e.g. GEE, OLS, Clustered OLS),
    that object is stored in ``fitted_model`` so you can call :meth:`model_summary`
    or use ``fitted_model.summary()`` for the full statsmodels output.

    Attributes:
        ate: Average treatment effect point estimate.
        p_value: P-value for the treatment effect.
        std_error: Standard error of the ATE.
        conf_int: Confidence interval for the ATE.
        fitted_model: The underlying fitted model when available (e.g. statsmodels
            GEE/OLS result). ``None`` for analyses that do not attach a model (e.g. Delta
            method, which uses closed-form formulas rather than a regression fit).

    Example:
        >>> analysis = GeeExperimentAnalysis(cluster_cols=["cluster"], target_col="target")
        >>> results = analysis.get_inference_results(df, alpha=0.05)
        >>> print(results.summary())       # ATE, p-value, CI, and model table if available
        >>> print(results.model_summary()) # Full statsmodels regression table
    """

    ate: float
    p_value: float
    std_error: float
    conf_int: ConfidenceInterval
    fitted_model: Optional[Any] = None

    def __str__(self) -> str:
        """
        Usage:
        ```python
        from cluster_experiments import ConfidenceInterval, InferenceResults
        ci = ConfidenceInterval(lower=0.1, upper=0.5, alpha=0.05)
        results = InferenceResults(ate=0.2, p_value=0.03, std_error=0.1, conf_int=ci)
        print(results)
        ```
        """
        return (
            f"ATE={self.ate:.4f}, p_value={self.p_value:.4f}, "
            f"std_error={self.std_error:.4f}, CI={self.conf_int}"
        )

    def model_summary(self) -> Optional[str]:
        """
        Return the full model fit summary when available (e.g. statsmodels
        regression table). Returns None if no fitted model was attached.

        Use this when you have run :meth:`get_inference_results` and want the
        same output you would get from the underlying model's ``.summary()``
        (e.g. coefficient table, standard errors, R-squared).
        """
        if self.fitted_model is None:
            return None
        if hasattr(self.fitted_model, "summary"):
            s = self.fitted_model.summary()
            if s is None:
                return None
            return str(s)
        return None

    def summary(self) -> str:
        """Return a summary of the inference results.

        Usage:
        ```python
        from cluster_experiments import ConfidenceInterval, InferenceResults
        ci = ConfidenceInterval(lower=0.1, upper=0.5, alpha=0.05)
        results = InferenceResults(ate=0.2, p_value=0.03, std_error=0.1, conf_int=ci)
        print(results.summary())
        ```
        """
        lines = [
            "Inference results",
            f"  ATE:        {self.ate:.6g}",
            f"  Std error:  {self.std_error:.6g}",
            f"  p-value:    {self.p_value:.6g}",
            f"  {self.conf_int.summary()}",
        ]
        model_str = self.model_summary()
        if model_str:
            lines.append("")
            lines.append("Model fit:")
            lines.append(model_str)
        return "\n".join(lines)


@dataclass
class RandomizationInferenceResults:
    """Results of a randomization (permutation) inference test.

    Randomization inference tests the sharp null hypothesis that the treatment
    has no effect on any unit. Under that null, every *legal* re-assignment of
    the treatment that the experimental design could have produced is equally
    plausible, so the observed statistic is compared against the statistics
    obtained by re-running the **same analysis** on **legally re-assigned**
    data.

    The reported p-value uses a finite number of randomizations ``R`` and the
    convention that the observed statistic is one extra reference draw, so the
    p-value can never be zero:

    * ``greater``: ``p = (1 + #{null >= observed}) / (1 + R)``
    * ``less``: ``p = (1 + #{null <= observed}) / (1 + R)``
    * ``two-sided``: ``p = (1 + #{|null| >= |observed|}) / (1 + R)``

    Ties (null statistics exactly equal to the observed statistic) **count as
    extreme**, and for one-sided hypotheses only the requested tail is used.
    Because ``R`` is finite, the p-value is bounded below by ``1 / (1 + R)``.

    Attributes:
        observed_statistic: Statistic computed on the observed data.
        p_value: Randomization-inference p-value (see the rule above).
        null_statistics: Statistics from every legal re-assignment.
        n_randomizations: Number of finite legal re-assignments ``R``.
        n_extreme: How many null statistics are at least as extreme as the
            observed one (ties included).
        n_ties: How many null statistics are exactly equal to the observed one.
        hypothesis: One of ``"two-sided"``, ``"less"``, ``"greater"``.
        alpha: Significance level stored for reporting purposes.
        statistic_name: Name of the statistic used (defaults to ``"ate"``).
        n_failed_randomizations: Number of re-assignments discarded because the
            statistic was undefined (e.g. a single arm) or non-finite.
    """

    observed_statistic: float
    p_value: float
    null_statistics: np.ndarray
    n_randomizations: int
    n_extreme: int
    n_ties: int
    hypothesis: str
    alpha: float = 0.05
    statistic_name: str = "ate"
    n_failed_randomizations: int = 0

    @property
    def ate(self) -> float:
        """Alias for the observed statistic (the analysis point estimate)."""
        return self.observed_statistic

    @property
    def p_value_rule(self) -> str:
        """Human-readable statement of the finite-randomization p-value rule."""
        tail = {
            HypothesisEntries.GREATER.value: "null statistics >= observed",
            HypothesisEntries.LESS.value: "null statistics <= observed",
            HypothesisEntries.TWO_SIDED.value: "|null statistics| >= |observed|",
        }[HypothesisEntries(self.hypothesis).value]
        return (
            f"p = (1 + #{{{tail}, ties included}}) / (1 + R) with "
            f"R = {self.n_randomizations} finite randomizations"
        )

    def summary(self) -> str:
        """Return a summary of the randomization-inference test."""
        return "\n".join(
            [
                "Randomization inference results",
                f"  Observed statistic: {self.observed_statistic:.6g}",
                f"  Hypothesis: {self.hypothesis}",
                f"  Randomizations: {self.n_randomizations}",
                f"  Extreme (ties included): {self.n_extreme}",
                f"  Ties: {self.n_ties}",
                f"  p-value: {self.p_value:.6g}",
                f"  Rule: {self.p_value_rule}",
            ]
        )


@dataclass
class RandomizationInferenceConfig:
    """Configuration for a randomization-inference test.

    It bundles the experimental design (through ``splitter``) with the finite
    number of randomizations, the random seed and the alternative hypothesis.

    Attributes:
        splitter: The configured :class:`~cluster_experiments.random_splitter.RandomSplitter`
            that describes the legal treatment assignment mechanism. It is used
            to generate every null re-assignment, so its cluster, stratification
            and washover rules are respected.
        n_randomizations: Number of finite legal re-assignments to draw.
        seed: Seed for the random source. When set, the whole test is
            reproducible and the caller's global random state is restored.
        hypothesis: One of ``"two-sided"``, ``"less"``, ``"greater"``. Defaults
            to the analysis' hypothesis.
        control: Name of the control treatment. Defaults to the only non-treated
            treatment of the splitter.
        statistic: Either ``"ate"`` (default) or a callable ``df -> float``.
        alpha: Significance level stored with the results.
        prepare_df: Optional callable applied to every frame (observed and
            re-assigned) right before the statistic is computed, after the rows
            are restricted to the control/treatment pair. Used by
            :class:`~cluster_experiments.inference.analysis_plan.AnalysisPlan`
            to restrict each test to a dimension slice while the splitter still
            re-assigns the full design.
    """

    splitter: Any
    n_randomizations: int = 1000
    seed: Optional[int] = None
    hypothesis: Optional[str] = None
    control: Optional[str] = None
    statistic: Optional[Any] = None
    alpha: Optional[float] = None
    prepare_df: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None

    def __post_init__(self):
        if self.splitter is None:
            raise ValueError("splitter is required for randomization inference")
        if not isinstance(self.n_randomizations, int) or self.n_randomizations < 1:
            raise ValueError(
                "n_randomizations must be a positive integer, "
                f"got {self.n_randomizations!r}"
            )
        if self.hypothesis is not None:
            HypothesisEntries(self.hypothesis)  # validates the value
        if not (
            self.statistic is None
            or self.statistic == "ate"
            or callable(self.statistic)
        ):
            raise ValueError(
                "statistic must be 'ate' or a callable, " f"got {self.statistic!r}"
            )
        if self.prepare_df is not None and not callable(self.prepare_df):
            raise ValueError("prepare_df must be a callable if provided")


@contextmanager
def temporary_random_seed(seed: Optional[int]):
    """Run a block with a fixed random source, restoring the caller's state.

    Splitters draw from both the global ``random`` module and NumPy's legacy
    global state (``pandas.DataFrame.sample`` and ``np.random.permutation``),
    so both are seeded and both are restored afterwards. This makes
    randomization inference reproducible without leaking global state.
    """
    if seed is None:
        yield
        return
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def randomization_p_value(
    null_statistics: np.ndarray, observed_statistic: float, hypothesis: str
) -> Tuple[float, int, int]:
    """Finite-randomization p-value, extreme count and tie count.

    The observed statistic is treated as one extra reference draw, hence the
    ``+ 1`` in both numerator and denominator. Ties (null statistics equal to
    the observed one up to floating-point tolerance) count as extreme, and for
    one-sided hypotheses only the requested tail is used.
    """
    null_statistics = np.asarray(null_statistics, dtype=float)
    observed_statistic = float(observed_statistic)
    # Equivalence tolerance so that numerically identical statistics (e.g. from
    # degenerate designs) are counted as ties and as extreme values.
    rtol = 1e-12 if observed_statistic == 0 else 1e-9

    def ties(values: np.ndarray) -> np.ndarray:
        return np.isclose(values, observed_statistic, rtol=rtol, atol=0.0)

    tie_mask = ties(null_statistics)
    if HypothesisEntries(hypothesis) == HypothesisEntries.GREATER:
        extreme_mask = (null_statistics >= observed_statistic) | tie_mask
    elif HypothesisEntries(hypothesis) == HypothesisEntries.LESS:
        extreme_mask = (null_statistics <= observed_statistic) | tie_mask
    elif HypothesisEntries(hypothesis) == HypothesisEntries.TWO_SIDED:
        extreme_mask = (
            np.abs(null_statistics) >= abs(observed_statistic)
        ) | np.isclose(
            np.abs(null_statistics), abs(observed_statistic), rtol=rtol, atol=0.0
        )
    else:
        raise ValueError(f"{hypothesis} is not a valid HypothesisEntries")

    n_extreme = int(extreme_mask.sum())
    n_ties = int(tie_mask.sum())
    n_randomizations = int(null_statistics.size)
    p_value = (1 + n_extreme) / (1 + n_randomizations)
    return p_value, n_extreme, n_ties


def prepare_observed_with_splitter(
    df: pd.DataFrame, splitter: "RandomSplitter"
) -> pd.DataFrame:
    """Apply the splitter's deterministic preparation to observed data.

    Falls back to a copy when the splitter does not implement
    ``prepare_observed_df`` (e.g. a user-defined splitter from an older
    version), which keeps randomization inference usable with third-party
    splitters.
    """
    prepare = getattr(splitter, "prepare_observed_df", None)
    if callable(prepare):
        return prepare(df)
    return df.copy()


def compute_randomization_statistics(
    df: pd.DataFrame,
    splitter: "RandomSplitter",
    statistic_fn: Callable[[pd.DataFrame], float],
    n_randomizations: int,
    seed: Optional[int],
    treatment_col: str = "treatment",
) -> Tuple[float, np.ndarray, int, Optional[str]]:
    """Observed statistic plus the null distribution from legal re-assignments.

    ``df`` must contain the design columns **and** the observed treatment in
    ``treatment_col``. The observed statistic is computed on ``df`` after the
    splitter's deterministic preparation. Null re-assignments are generated by
    the splitter from a copy of ``df`` with the observed treatment removed, so
    the design's own assignment mechanism (clusters, strata, washover) is the
    only thing that determines the labels. ``statistic_fn`` is applied unchanged
    to the observed data and to every re-assigned dataset.

    Returns ``(observed_statistic, null_statistics, n_failed, last_failure)``.
    Re-assignments whose statistic is undefined raise inside ``statistic_fn``
    and are counted as failures rather than aborting the whole test.
    """
    reassignment_input = df.drop(columns=[treatment_col])
    with temporary_random_seed(seed):
        observed_df = prepare_observed_with_splitter(df, splitter)
        observed_statistic = statistic_fn(observed_df)

        null_statistics: List[float] = []
        n_failed = 0
        last_failure: Optional[str] = None
        for _ in range(n_randomizations):
            reassigned_df = splitter.assign_treatment_df(reassignment_input)
            if len(reassigned_df) > len(df):
                raise ValueError(
                    "The splitter returned more rows than it received, so it is "
                    "not a legal one-assignment-per-row mechanism (RepeatedSampler "
                    f"and similar splitters cannot be used): {len(reassigned_df)} "
                    f"rows for {len(df)} inputs."
                )
            try:
                null_statistics.append(statistic_fn(reassigned_df))
            except (ValueError, AssertionError, ZeroDivisionError) as exc:
                n_failed += 1
                last_failure = str(exc)

    return (
        observed_statistic,
        np.asarray(null_statistics, dtype=float),
        n_failed,
        last_failure,
    )


class ExperimentAnalysis(ABC):
    """
    Abstract class to run the analysis of a given experiment

    In order to create your own ExperimentAnalysis,
    you should create a derived class that implements the analysis_pvalue method.

    It can also be used as a component of the PowerAnalysis class.

    Arguments:
        cluster_cols: list of columns to use as clusters
        target_col: name of the column containing the variable to measure
        treatment_col: name of the column containing the treatment variable
        treatment: name of the treatment to use as the treated group
        covariates: list of columns to use as covariates
        hypothesis: one of "two-sided", "less", "greater" indicating the alternative hypothesis

    """

    def __init__(
        self,
        cluster_cols: List[str],
        target_col: str = "target",
        treatment_col: str = "treatment",
        treatment: str = "B",
        covariates: Optional[List[str]] = None,
        hypothesis: str = "two-sided",
        add_covariate_interaction: bool = False,
    ):
        self.target_col = target_col
        self.treatment = treatment
        self.treatment_col = treatment_col
        self.cluster_cols = cluster_cols
        self.covariates = covariates or []
        self.hypothesis = hypothesis
        self.add_covariate_interaction = add_covariate_interaction

    def __repr__(self) -> str:
        """
        Usage:
        ```python
        from cluster_experiments import GeeExperimentAnalysis
        a = GeeExperimentAnalysis(cluster_cols=["cluster"])
        print(repr(a))
        ```
        """
        return (
            f"{type(self).__name__}(cluster_cols={self.cluster_cols!r}, "
            f"target_col={self.target_col!r}, treatment_col={self.treatment_col!r}, "
            f"treatment={self.treatment!r}, hypothesis={self.hypothesis!r})"
        )

    def __str__(self) -> str:
        """
        Usage:
        ```python
        from cluster_experiments import GeeExperimentAnalysis
        a = GeeExperimentAnalysis(cluster_cols=["cluster"])
        print(a)
        ```
        """
        return (
            f"{type(self).__name__}: cluster_cols={self.cluster_cols}, "
            f"target={self.target_col}, treatment={self.treatment}"
        )

    def _get_cluster_column(self, df: pd.DataFrame) -> pd.Series:
        """Paste all strings of cluster_cols in one single column"""
        df = df.copy()
        return df[self.cluster_cols].astype(str).sum(axis=1)

    def _create_binary_treatment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms treatment column into 0 - 1 column"""
        df = df.copy()
        df[self.treatment_col] = (df[self.treatment_col] == self.treatment).astype(int)
        return df

    def _add_interaction_covariates(self, df: pd.DataFrame) -> pd.DataFrame:
        """For each covariate, adds a column with treatment * (x - mean(x))
        This is used to build a more efficient estimator of the ATE

        Args
        ----
            df (pd.DataFrame): input data frame

        Returns
        -------
            pd.DataFrame: data frame with additional columns

        """
        df = df.copy()
        if self.covariates is None:
            return df

        for covariate in self.covariates:
            df[f"__{covariate}__interaction"] = (
                df[covariate] - df[covariate].mean()
            ) * df[self.treatment_col]
        return df

    @property
    def covariates_list(self) -> List[str]:
        if len(self.covariates) == 0:
            # simple case, no covariates
            return []

        if not self.add_covariate_interaction:
            # second case: covariates but not interaction
            return self.covariates

        # third case: covariates and interaction
        return self.covariates + [
            f"__{covariate}__interaction" for covariate in self.covariates
        ]

    @property
    def formula(self):
        if len(self.covariates) == 0:
            # simple case, no covariates
            return f"{self.target_col} ~ {self.treatment_col}"

        if not self.add_covariate_interaction:
            # second case: covariates but not interaction
            return f"{self.target_col} ~ {self.treatment_col} + {' + '.join(self.covariates)}"

        # third case: covariates and interaction
        return f"{self.target_col} ~ {self.treatment_col} + {' + '.join(self.covariates)} + {' + '.join([f'__{covariate}__interaction' for covariate in self.covariates])}"

    @abstractmethod
    def analysis_pvalue(
        self,
        df: pd.DataFrame,
        verbose: bool = False,
    ) -> float:
        """
        Returns the p-value of the analysis. Expects treatment to be 0-1 variable
        Arguments:
            df: dataframe containing the data to analyze
            verbose (Optional): bool, prints the regression summary if True
        """

    def analysis_point_estimate(
        self,
        df: pd.DataFrame,
        verbose: bool = False,
    ) -> float:
        """
        Returns the point estimate of the analysis. Expects treatment to be 0-1 variable
        Arguments:
            df: dataframe containing the data to analyze
            verbose (Optional): bool, prints the regression summary if True
        """
        raise NotImplementedError("Point estimate not implemented for this analysis")

    def analysis_standard_error(
        self,
        df: pd.DataFrame,
        verbose: bool = False,
    ) -> float:
        """
        Returns the standard error of the analysis. Expects treatment to be 0-1 variable
        Arguments:
            df: dataframe containing the data to analyze
            verbose (Optional): bool, prints the regression summary if True
        """
        raise NotImplementedError("Standard error not implemented for this analysis")

    def analysis_confidence_interval(
        self,
        df: pd.DataFrame,
        alpha: float,
        verbose: bool = False,
    ) -> ConfidenceInterval:
        """
        Returns the confidence interval of the analysis. Expects treatment to be 0-1 variable
        Arguments:
            df: dataframe containing the data to analyze
            alpha: significance level
            verbose (Optional): bool, prints the regression summary if True
        """
        raise NotImplementedError(
            "Confidence Interval not implemented for this analysis"
        )

    def analysis_inference_results(
        self,
        df: pd.DataFrame,
        alpha: float,
        verbose: bool = False,
    ) -> InferenceResults:
        """
        Returns the InferenceResults object of the analysis. Expects treatment to be 0-1 variable
        Arguments:
            df: dataframe containing the data to analyze
            alpha: significance level
            verbose (Optional): bool, prints the regression summary if True
        """
        raise NotImplementedError(
            "Inference results are not implemented for this analysis"
        )

    def _data_checks(self, df: pd.DataFrame) -> None:
        """Checks that the data is correct"""
        if df[self.target_col].isnull().any():
            raise ValueError(
                f"There are null values in outcome column {self.treatment_col}"
            )

        if not is_numeric_dtype(df[self.target_col]):
            raise ValueError(
                f"Outcome column {self.target_col} should be numeric and not {df[self.target_col].dtype}"
            )

    def get_pvalue(self, df: pd.DataFrame) -> float:
        """Returns the p-value of the analysis

        Arguments:
            df: dataframe containing the data to analyze
        """
        df = df.copy()
        df = self._create_binary_treatment(df)
        self._data_checks(df=df)
        return self.analysis_pvalue(df)

    def get_point_estimate(self, df: pd.DataFrame) -> float:
        """Returns the point estimate of the analysis

        Arguments:
            df: dataframe containing the data to analyze
        """
        df = df.copy()
        df = self._create_binary_treatment(df)
        self._data_checks(df=df)
        return self.analysis_point_estimate(df)

    def get_standard_error(self, df: pd.DataFrame) -> float:
        """Returns the standard error of the analysis

        Arguments:
            df: dataframe containing the data to analyze
        """
        df = df.copy()
        df = self._create_binary_treatment(df)
        self._data_checks(df=df)
        return self.analysis_standard_error(df)

    def get_confidence_interval(
        self, df: pd.DataFrame, alpha: float
    ) -> ConfidenceInterval:
        """Returns the confidence interval of the analysis

        Arguments:
            df: dataframe containing the data to analyze
            alpha: significance level
        """
        df = df.copy()
        df = self._create_binary_treatment(df)
        self._data_checks(df=df)
        return self.analysis_confidence_interval(df, alpha)

    def get_inference_results(self, df: pd.DataFrame, alpha: float) -> InferenceResults:
        """Returns the inference results of the analysis for a single dataset.

        Use this when you want full results (ATE, p-value, standard error, CI)
        and optionally the underlying model fit (e.g. statsmodels regression
        table) via ``results.summary()`` or ``results.model_summary()``.
        Power analysis does not use this method; it uses :meth:`get_pvalue` and
        :meth:`get_point_estimate` over many simulations.

        Arguments:
            df: dataframe containing the data to analyze
            alpha: significance level

        Returns:
            InferenceResults with ate, p_value, std_error, conf_int, and
            fitted_model when the analysis attaches one (GEE, OLS, Delta).
        """
        df = df.copy()
        df = self._create_binary_treatment(df)
        self._data_checks(df=df)
        return self.analysis_inference_results(df, alpha)

    def _get_randomization_control(
        self, splitter: "RandomSplitter", control: Optional[str]
    ) -> str:
        """Resolve the control label used to build the binary contrast."""
        if control is not None:
            if control == self.treatment:
                raise ValueError(
                    "control and treatment must be different for randomization "
                    f"inference, both are {control!r}"
                )
            return control
        treatments = list(getattr(splitter, "treatments", None) or [])
        candidates = [t for t in treatments if t != self.treatment]
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise ValueError(
                f"Could not infer a control different from treatment "
                f"{self.treatment!r} "
                f"from splitter treatments {treatments}. Pass control explicitly."
            )
        raise ValueError(
            f"Could not infer a unique control from splitter treatments {treatments}; "
            f"pass control explicitly."
        )

    def _validate_randomization_arms(
        self, df: pd.DataFrame, treatments: List[str], control: str
    ) -> None:
        """Check that the splitter can reproduce the observed experiment arms.

        Randomization inference compares the observed contrast with contrasts
        from designs the splitter can generate. If the observed data contains
        arms outside ``splitter.treatments``, the two sides would describe
        different experiments, so it is rejected rather than silently producing
        an invalid p-value.
        """
        if not treatments:
            return
        observed_arms = set(df[self.treatment_col].dropna().unique())
        unexpected = observed_arms - set(treatments)
        if unexpected:
            raise ValueError(
                f"Observed treatment labels {sorted(unexpected)} are not among the "
                f"splitter treatments {treatments}. Randomization inference can only "
                "compare against re-assignments the design could have produced."
            )
        if control not in treatments:
            raise ValueError(
                f"control {control!r} is not one of the splitter treatments "
                f"{treatments}; the observed contrast is not a design arm."
            )

    def _randomization_base_df(
        self, df: pd.DataFrame, splitter: "RandomSplitter"
    ) -> pd.DataFrame:
        """Build the data both the observed and null statistics start from.

        All design rows are kept, because the legal assignment mechanism acts on
        the whole design (a cluster or a time period may belong to a third arm
        in a multi-arm experiment). The control/treatment contrast is applied
        later, identically, by ``_randomization_transform`` on both sides.

        When the splitter already stored the original time column, the raw time
        column is restored so running the splitter again is idempotent.
        """
        base_df = df.reset_index(drop=True)
        time_col = getattr(splitter, "time_col", None)
        if time_col is not None:
            original_time_col = _original_time_column(time_col)
            if original_time_col in base_df.columns:
                base_df = base_df.assign(**{time_col: base_df[original_time_col]})
        return base_df

    def _randomization_transform(
        self,
        frame: pd.DataFrame,
        control: str,
        config: RandomizationInferenceConfig,
    ) -> pd.DataFrame:
        """Restrict to the contrast rows and apply the optional ``prepare_df``.

        This is the single place where "data preparation" happens, so the
        observed statistic and every null statistic are computed on data built
        by the same steps.
        """
        prepared = frame.loc[
            frame[self.treatment_col].isin([control, self.treatment])
        ].reset_index(drop=True)
        if config.prepare_df is not None:
            prepared = config.prepare_df(prepared)
        return prepared

    def _randomization_statistic(
        self,
        prepared_df: pd.DataFrame,
        statistic: Optional[Callable[[pd.DataFrame], float]],
    ) -> float:
        """Compute one statistic with the analysis' own definition.

        The default statistic is :meth:`get_point_estimate`, i.e. exactly the
        point estimate the analysis reports on the observed data. A custom
        callable receives the same filtered dataframe. ``NaN`` and infinite
        values are treated as invalid re-assignments by the caller.
        """
        if statistic is None or statistic == "ate":
            return float(self.get_point_estimate(prepared_df))
        return float(statistic(prepared_df.copy()))

    def get_randomization_inference(
        self,
        df: pd.DataFrame,
        config: RandomizationInferenceConfig,
        alpha: Optional[float] = None,
    ) -> RandomizationInferenceResults:
        """Randomization (permutation) inference using the design's splitter.

        The observed statistic is computed on the data prepared by the same
        splitter rules (time truncation for switchback, washover row dropping),
        and every null statistic is computed by re-running the **same analysis
        definition** on data re-assigned by the splitter. Because the splitter
        is reused, cluster-level assignment, stratification and washover rules
        all apply to the null distribution: unit-level labels are never shuffled
        independently of their cluster.

        The p-value uses a finite number of randomizations and never reaches
        zero:

        * ``greater``: ``(1 + #{null >= observed}) / (1 + R)``
        * ``less``: ``(1 + #{null <= observed}) / (1 + R)``
        * ``two-sided``: ``(1 + #{|null| >= |observed|}) / (1 + R)``

        Ties count as extreme. Re-assignments that yield a degenerate statistic
        (e.g. a single arm) are discarded and reported in
        ``n_failed_randomizations``; ``R`` is the number of valid ones.

        Arguments:
            df: dataframe with the observed treatment and outcome (raw data, or
                data already processed by the same splitter; processing is
                idempotent).
            config: randomization-inference configuration, including the
                splitter that defines the legal assignment mechanism.
            alpha: significance level stored with the results (defaults to the
                config's ``alpha``, then ``0.05``).

        Returns:
            RandomizationInferenceResults with the observed statistic, the null
            distribution and the finite-randomization p-value.

        Usage:
        ```python
        import pandas as pd

        from cluster_experiments import (
            ClusteredOLSAnalysis,
            ClusteredSplitter,
            RandomizationInferenceConfig,
        )

        df = pd.DataFrame(
            {
                "cluster": [f"c{i}" for i in range(24) for _ in range(4)],
                "treatment": [
                    "A" if i % 2 == 0 else "B" for i in range(24) for _ in range(4)
                ],
                "target": [
                    1.0 + (0.3 if i % 2 == 0 else 0.0) + 0.1 * j
                    for i in range(24)
                    for j in range(4)
                ],
            }
        )
        splitter = ClusteredSplitter(
            cluster_cols=["cluster"], treatment_col="treatment", treatments=["A", "B"]
        )
        analysis = ClusteredOLSAnalysis(
            cluster_cols=["cluster"], target_col="target", treatment="B"
        )
        results = analysis.get_randomization_inference(
            df,
            RandomizationInferenceConfig(
                splitter=splitter, n_randomizations=100, seed=0
            ),
        )
        print(results.p_value)
        ```
        """
        splitter = config.splitter
        if self.treatment_col not in df.columns:
            raise ValueError(
                f"Randomization inference needs the observed assignment in column "
                f"{self.treatment_col!r}. Assign the experiment first (for example "
                "with `splitter.assign_treatment_df`) and pass that data."
            )
        if getattr(splitter, "treatment_col", self.treatment_col) != self.treatment_col:
            raise ValueError(
                "The splitter must assign the same column the analysis reads: "
                f"splitter.treatment_col={getattr(splitter, 'treatment_col', None)!r} "
                f"but analysis.treatment_col={self.treatment_col!r}."
            )

        hypothesis = config.hypothesis or self.hypothesis
        HypothesisEntries(hypothesis)  # validate
        treatments = list(getattr(splitter, "treatments", None) or [])
        if treatments and self.treatment not in treatments:
            raise ValueError(
                f"analysis treatment {self.treatment!r} is not one of the splitter "
                f"treatments {treatments}; the null re-assignments would never "
                "produce the treatment label under test"
            )
        control = self._get_randomization_control(splitter, config.control)
        # The design must be able to reproduce the observed assignment's arms;
        # otherwise the observed rows under the contrast and the re-assigned
        # rows would describe different experiments.
        self._validate_randomization_arms(df, treatments, control)
        resolved_alpha = (
            alpha
            if alpha is not None
            else (config.alpha if config.alpha is not None else 0.05)
        )
        statistic = config.statistic
        statistic_name = "ate" if statistic in (None, "ate") else "custom"

        base_df = self._randomization_base_df(df, splitter=splitter)

        def compute_statistic(frame: pd.DataFrame) -> float:
            prepared = self._randomization_transform(
                frame, control=control, config=config
            )
            if prepared[self.treatment_col].nunique() < 2:
                raise ValueError(
                    "re-assignment produced a single arm; statistic undefined"
                )
            try:
                value = self._randomization_statistic(prepared, statistic)
            except NotImplementedError as exc:
                raise ValueError(
                    "This analysis does not implement a point estimate, which is "
                    "the default randomization-inference statistic. Pass "
                    "`statistic` as a callable (df -> float)."
                ) from exc
            if not np.isfinite(value):
                raise ValueError("re-assignment produced a non-finite statistic")
            return value

        observed_statistic, null_array, n_failed, last_failure = (
            compute_randomization_statistics(
                base_df,
                splitter=splitter,
                statistic_fn=compute_statistic,
                n_randomizations=config.n_randomizations,
                seed=config.seed,
                treatment_col=self.treatment_col,
            )
        )

        n_randomizations = int(null_array.size)
        if n_randomizations == 0:
            raise ValueError(
                "No valid randomization was produced by the splitter; "
                f"{n_failed} attempts failed. Last error: {last_failure}"
            )
        if n_failed:
            warnings.warn(
                f"{n_failed} of {config.n_randomizations} legal re-assignments "
                f"produced an undefined statistic and were discarded. Last error: "
                f"{last_failure}",
                stacklevel=2,
            )

        p_value, n_extreme, n_ties = randomization_p_value(
            null_array, observed_statistic, hypothesis
        )

        return RandomizationInferenceResults(
            observed_statistic=observed_statistic,
            p_value=p_value,
            null_statistics=null_array,
            n_randomizations=n_randomizations,
            n_extreme=n_extreme,
            n_ties=n_ties,
            hypothesis=hypothesis,
            alpha=resolved_alpha,
            statistic_name=statistic_name,
            n_failed_randomizations=n_failed,
        )

    def pvalue_based_on_hypothesis(
        self, model_result: RegressionResultsProtocol
    ) -> float:  # todo add typehint statsmodels result
        """Returns the p-value of the analysis
        Arguments:
            model_result: statsmodels result object
            verbose (Optional): bool, prints the regression summary if True

        """
        treatment_effect = model_result.params[self.treatment_col]
        p_value = model_result.pvalues[self.treatment_col]

        if HypothesisEntries(self.hypothesis) == HypothesisEntries.LESS:
            return p_value / 2 if treatment_effect <= 0 else 1 - p_value / 2
        if HypothesisEntries(self.hypothesis) == HypothesisEntries.GREATER:
            return p_value / 2 if treatment_effect >= 0 else 1 - p_value / 2
        if HypothesisEntries(self.hypothesis) == HypothesisEntries.TWO_SIDED:
            return p_value
        raise ValueError(f"{self.hypothesis} is not a valid HypothesisEntries")

    def _split_pre_experiment_df(self, df: pd.DataFrame):
        raise NotImplementedError(
            "This method should be implemented in the child class"
        )

    @classmethod
    def from_config(cls, config):
        """Creates an ExperimentAnalysis object from a PowerConfig object"""
        return cls(
            cluster_cols=config.cluster_cols,
            target_col=config.target_col,
            treatment_col=config.treatment_col,
            treatment=config.treatment,
            covariates=config.covariates,
            hypothesis=config.hypothesis,
            add_covariate_interaction=config.add_covariate_interaction,
        )


class GeeExperimentAnalysis(ExperimentAnalysis):
    """
    Class to run GEE clustered analysis

    Arguments:
        cluster_cols: list of columns to use as clusters
        target_col: name of the column containing the variable to measure
        treatment_col: name of the column containing the treatment variable
        treatment: name of the treatment to use as the treated group
        covariates: list of columns to use as covariates
        hypothesis: one of "two-sided", "less", "greater" indicating the alternative hypothesis

    Usage:

    ```python
    from cluster_experiments.experiment_analysis import GeeExperimentAnalysis
    import pandas as pd

    df = pd.DataFrame({
        'x': [1, 2, 3, 0, 0, 1],
        'treatment': ["A"] * 3 + ["B"] * 3,
        'cluster': [1] * 6,
    })

    GeeExperimentAnalysis(
        cluster_cols=['cluster'],
        target_col='x',
    ).get_pvalue(df)
    ```
    """

    def __init__(
        self,
        cluster_cols: List[str],
        target_col: str = "target",
        treatment_col: str = "treatment",
        treatment: str = "B",
        covariates: Optional[List[str]] = None,
        hypothesis: str = "two-sided",
        add_covariate_interaction: bool = False,
    ):
        super().__init__(
            target_col=target_col,
            treatment_col=treatment_col,
            cluster_cols=cluster_cols,
            treatment=treatment,
            covariates=covariates,
            hypothesis=hypothesis,
            add_covariate_interaction=add_covariate_interaction,
        )
        self.fam = sm.families.Gaussian()
        self.va = sm.cov_struct.Exchangeable()

    def fit_gee(self, df: pd.DataFrame) -> sm.GEE:
        """Returns the fitted GEE model"""
        if self.add_covariate_interaction:
            df = self._add_interaction_covariates(df)
        return sm.GEE.from_formula(
            self.formula,
            data=df,
            groups=self._get_cluster_column(df),
            family=self.fam,
            cov_struct=self.va,
        ).fit()

    def analysis_pvalue(self, df: pd.DataFrame, verbose: bool = False) -> float:
        """Returns the p-value of the analysis
        Arguments:
            df: dataframe containing the data to analyze
            verbose (Optional): bool, prints the regression summary if True
        """
        results_gee = self.fit_gee(df)
        if verbose:
            print(results_gee.summary())

        p_value = self.pvalue_based_on_hypothesis(results_gee)
        return p_value

    def analysis_point_estimate(self, df: pd.DataFrame, verbose: bool = False) -> float:
        """Returns the point estimate of the analysis
        Arguments:
            df: dataframe containing the data to analyze
            verbose (Optional): bool, prints the regression summary if True
        """
        results_gee = self.fit_gee(df)
        return results_gee.params[self.treatment_col]

    def analysis_standard_error(self, df: pd.DataFrame, verbose: bool = False) -> float:
        """Returns the standard error of the analysis
        Arguments:
            df: dataframe containing the data to analyze
            verbose (Optional): bool, prints the regression summary if True
        """
        results_gee = self.fit_gee(df)
        return results_gee.bse[self.treatment_col]

    def analysis_confidence_interval(
        self, df: pd.DataFrame, alpha: float, verbose: bool = False
    ) -> ConfidenceInterval:
        """Returns the confidence interval of the analysis
        Arguments:
            df: dataframe containing the data to analyze
            alpha: significance level
            verbose (Optional): bool, prints the regression summary if True
        """
        results_gee = self.fit_gee(df)
        # Extract the confidence interval for the treatment column
        conf_int_df = results_gee.conf_int(alpha=alpha)
        lower_bound, upper_bound = conf_int_df.loc[self.treatment_col]

        if verbose:
            print(results_gee.summary())

        # Return the confidence interval
        return ConfidenceInterval(lower=lower_bound, upper=upper_bound, alpha=alpha)

    def analysis_inference_results(
        self, df: pd.DataFrame, alpha: float, verbose: bool = False
    ) -> InferenceResults:
        """Returns the inference results of the analysis
        Arguments:
            df: dataframe containing the data to analyze
            alpha: significance level
            verbose (Optional): bool, prints the regression summary if True
        """
        results_gee = self.fit_gee(df)

        std_error = results_gee.bse[self.treatment_col]
        ate = results_gee.params[self.treatment_col]
        p_value = self.pvalue_based_on_hypothesis(results_gee)

        # Extract the confidence interval for the treatment column
        conf_int_df = results_gee.conf_int(alpha=alpha)
        lower_bound, upper_bound = conf_int_df.loc[self.treatment_col]

        if verbose:
            print(results_gee.summary())

        # Return the confidence interval
        return InferenceResults(
            ate=ate,
            p_value=p_value,
            std_error=std_error,
            conf_int=ConfidenceInterval(
                lower=lower_bound, upper=upper_bound, alpha=alpha
            ),
            fitted_model=results_gee,
        )


class TTestClusteredAnalysis(ExperimentAnalysis):
    """
    Class to run T-test analysis on aggregated data

    Arguments:
        cluster_cols: list of columns to use as clusters
        target_col: name of the column containing the variable to measure
        treatment_col: name of the column containing the treatment variable
        treatment: name of the treatment to use as the treated group
        hypothesis: one of "two-sided", "less", "greater" indicating the alternative hypothesis

    Usage:

    ```python
    from cluster_experiments.experiment_analysis import TTestClusteredAnalysis
    import pandas as pd

    df = pd.DataFrame({
        'x': [1, 2, 3, 4, 0, 0, 1, 1],
        'treatment': ["A", "B", "A", "B"] * 2,
        'cluster': [1, 2, 3, 4, 1, 2, 3, 4],
    })

    TTestClusteredAnalysis(
        cluster_cols=['cluster'],
        target_col='x',
    ).get_pvalue(df)
    ```
    """

    def __init__(
        self,
        cluster_cols: List[str],
        target_col: str = "target",
        treatment_col: str = "treatment",
        treatment: str = "B",
        hypothesis: str = "two-sided",
    ):
        self.target_col = target_col
        self.treatment = treatment
        self.treatment_col = treatment_col
        self.cluster_cols = cluster_cols
        self.hypothesis = hypothesis

    def analysis_pvalue(self, df: pd.DataFrame, verbose: bool = False) -> float:
        """Returns the p-value of the analysis
        Arguments:
            df: dataframe containing the data to analyze
            verbose (Optional): bool, prints the regression summary if True
        """

        df_grouped = df.groupby(
            self.cluster_cols + [self.treatment_col], as_index=False
        )[self.target_col].mean()

        treatment_data = df_grouped.query(f"{self.treatment_col} == 1")[self.target_col]
        control_data = df_grouped.query(f"{self.treatment_col} == 0")[self.target_col]
        assert len(treatment_data), "treatment data should have more than 1 cluster"
        assert len(control_data), "control data should have more than 1 cluster"
        t_test_results = ttest_ind(
            treatment_data, control_data, equal_var=False, alternative=self.hypothesis
        )
        return t_test_results.pvalue

    def analysis_point_estimate(self, df: pd.DataFrame, verbose: bool = False) -> float:
        """Returns the difference in cluster-level means (treatment - control).

        This is the natural statistic for randomization inference on this
        analysis: it uses the same cluster aggregation as the t-test.

        Arguments:
            df: dataframe containing the data to analyze
            verbose (Optional): bool, prints the cluster table if True
        """
        df_grouped = df.groupby(
            self.cluster_cols + [self.treatment_col], as_index=False
        )[self.target_col].mean()

        treatment_data = df_grouped.query(f"{self.treatment_col} == 1")[self.target_col]
        control_data = df_grouped.query(f"{self.treatment_col} == 0")[self.target_col]
        assert len(treatment_data), "treatment data should have more than 1 cluster"
        assert len(control_data), "control data should have more than 1 cluster"
        if verbose:
            print(f"cluster-level means (treatment):\n{treatment_data.describe()}")
            print(f"cluster-level means (control):\n{control_data.describe()}")
        return float(treatment_data.mean() - control_data.mean())

    @classmethod
    def from_config(cls, config):
        """Creates a TTestClusteredAnalysis object from a PowerConfig object"""
        return cls(
            cluster_cols=config.cluster_cols,
            target_col=config.target_col,
            treatment_col=config.treatment_col,
            treatment=config.treatment,
            hypothesis=config.hypothesis,
        )


class PairedTTestClusteredAnalysis(ExperimentAnalysis):
    """
    Class to run paired T-test analysis on aggregated data

    Arguments:
        cluster_cols: list of columns to use as clusters
        target_col: name of the column containing the variable to measure
        treatment_col: name of the column containing the treatment variable
        treatment: name of the treatment to use as the treated group
        strata_cols: list of index columns for paired t test. Should be a subset or equal to cluster_cols
        hypothesis: one of "two-sided", "less", "greater" indicating the alternative hypothesis

    Usage:

    ```python
    from cluster_experiments.experiment_analysis import PairedTTestClusteredAnalysis
    import pandas as pd

    df = pd.DataFrame({
        'x': [1, 2, 3, 4, 0, 0, 1, 1],
        'treatment': ["A", "B", "A", "B"] * 2,
        'cluster': [1, 2, 3, 4, 1, 2, 3, 4],
    })

    PairedTTestClusteredAnalysis(
        cluster_cols=['cluster'],
        strata_cols=['cluster'],
        target_col='x',
    ).get_pvalue(df)
    ```
    """

    def __init__(
        self,
        cluster_cols: List[str],
        strata_cols: List[str],
        target_col: str = "target",
        treatment_col: str = "treatment",
        treatment: str = "B",
        hypothesis: str = "two-sided",
    ):
        self.strata_cols = strata_cols
        self.target_col = target_col
        self.treatment = treatment
        self.treatment_col = treatment_col
        self.cluster_cols = cluster_cols
        self.hypothesis = hypothesis

    def _preprocessing(self, df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        df_grouped = df.groupby(
            self.cluster_cols + [self.treatment_col], as_index=False
        )[self.target_col].mean()

        n_control = df_grouped[self.treatment_col].value_counts()[0]
        n_treatment = df_grouped[self.treatment_col].value_counts()[1]

        if n_control != n_treatment:
            logging.warning(
                f"groups don't have same number of observations, {n_treatment =} and  {n_control =}"
            )

        assert all(
            [x in self.cluster_cols for x in self.strata_cols]
        ), f"strata should be a subset or equal to cluster_cols ({self.cluster_cols = }, {self.strata_cols = })"

        df_pivot = df_grouped.pivot_table(
            columns=self.treatment_col,
            index=self.strata_cols,
            values=self.target_col,
        )

        if df_pivot.isna().sum().sum() > 0:
            logging.warning(
                f"There are missing pairs for some clusters, removing the lonely ones: {df_pivot[df_pivot.isna().any(axis=1)].to_dict()}"
            )

        if verbose:
            print(f"performing paired t test in this data \n {df_pivot} \n")

        df_pivot = df_pivot.dropna()

        return df_pivot

    def analysis_pvalue(self, df: pd.DataFrame, verbose: bool = False) -> float:
        """Returns the p-value of the analysis
        Arguments:
            df: dataframe containing the data to analyze
            verbose (Optional): bool, prints the extra info if True
        """
        assert (
            type(self.cluster_cols) is list
        ), "cluster_cols needs to be a list of strings (even with one element)"
        assert (
            type(self.strata_cols) is list
        ), "strata_cols needs to be a list of strings (even with one element)"

        df_pivot = self._preprocessing(df=df)

        t_test_results = ttest_rel(
            df_pivot.iloc[:, 0], df_pivot.iloc[:, 1], alternative=self.hypothesis
        )

        if verbose:
            print(f"paired t test results: \n {t_test_results} \n")

        return t_test_results.pvalue

    def analysis_point_estimate(self, df: pd.DataFrame, verbose: bool = False) -> float:
        """Returns the mean paired difference used by the paired t-test.

        Arguments:
            df: dataframe containing the data to analyze
            verbose (Optional): bool, prints the paired pivot table if True
        """
        df_pivot = self._preprocessing(df=df, verbose=verbose)
        return float((df_pivot.iloc[:, 1] - df_pivot.iloc[:, 0]).mean())

    @classmethod
    def from_config(cls, config):
        """Creates a PairedTTestClusteredAnalysis object from a PowerConfig object"""
        return cls(
            cluster_cols=config.cluster_cols,
            target_col=config.target_col,
            treatment_col=config.treatment_col,
            treatment=config.treatment,
            strata_cols=config.strata_cols,
            hypothesis=config.hypothesis,
        )


class OLSAnalysis(ExperimentAnalysis):
    """
    Class to run OLS analysis

    Arguments:
        target_col: name of the column containing the variable to measure
        treatment_col: name of the column containing the treatment variable
        treatment: name of the treatment to use as the treated group
        covariates: list of columns to use as covariates
        hypothesis: one of "two-sided", "less", "greater" indicating the alternative hypothesis
        cov_type: one of "nonrobust", "fixed scale", "HC0", "HC1", "HC2", "HC3", "HAC", "hac-panel", "hac-groupsum", "cluster"

    Usage:

    ```python
    from cluster_experiments.experiment_analysis import OLSAnalysis
    import pandas as pd

    df = pd.DataFrame({
        'x': [1, 2, 3, 0, 0, 1],
        'treatment': ["A"] * 3 + ["B"] * 3,
    })

    OLSAnalysis(
        target_col='x',
    ).get_pvalue(df)
    ```
    """

    def __init__(
        self,
        target_col: str = "target",
        treatment_col: str = "treatment",
        treatment: str = "B",
        covariates: Optional[List[str]] = None,
        hypothesis: str = "two-sided",
        cov_type: Optional[
            Literal[
                "nonrobust",
                "fixed scale",
                "HC0",
                "HC1",
                "HC2",
                "HC3",
                "HAC",
                "hac-panel",
                "hac-groupsum",
                "cluster",
            ]
        ] = None,
        add_covariate_interaction: bool = False,
        relative_effect: bool = False,
    ):
        self.target_col = target_col
        self.treatment = treatment
        self.treatment_col = treatment_col
        self.covariates = covariates or []
        self.hypothesis = hypothesis
        self.cov_type: Literal[
            "nonrobust",
            "fixed scale",
            "HC0",
            "HC1",
            "HC2",
            "HC3",
            "HAC",
            "hac-panel",
            "hac-groupsum",
            "cluster",
        ] = (
            "HC3" if cov_type is None else cov_type
        )
        self.add_covariate_interaction = add_covariate_interaction
        self.relative_effect = relative_effect

    def fit_ols(self, df: pd.DataFrame) -> RegressionResultsProtocol:
        """Returns the fitted OLS model"""
        if self.add_covariate_interaction:
            df = self._add_interaction_covariates(df)

        ols_fit = sm.OLS.from_formula(self.formula, data=df).fit(cov_type=self.cov_type)

        # create point estimate, pvalue and std error transformation in case of relative effects
        if self.relative_effect:
            relative_ols_fit = LiftRegressionTransformer(
                treatment_col=self.treatment_col
            )
            relative_ols_fit.fit(
                ols=ols_fit, df=df, covariate_cols=self.covariates_list
            )
            return relative_ols_fit

        return ols_fit

    def analysis_pvalue(self, df: pd.DataFrame, verbose: bool = False) -> float:
        """Returns the p-value of the analysis
        Arguments:
            df: dataframe containing the data to analyze
            verbose (Optional): bool, prints the regression summary if True
        """
        results_ols = self.fit_ols(df=df)
        if verbose:
            print(results_ols.summary())

        p_value = self.pvalue_based_on_hypothesis(results_ols)
        return p_value

    def analysis_point_estimate(self, df: pd.DataFrame, verbose: bool = False) -> float:
        """Returns the point estimate of the analysis
        Arguments:
            df: dataframe containing the data to analyze
            verbose (Optional): bool, prints the regression summary if True
        """
        results_ols = self.fit_ols(df=df)
        return results_ols.params[self.treatment_col]

    def analysis_standard_error(self, df: pd.DataFrame, verbose: bool = False) -> float:
        """Returns the standard error of the analysis
        Arguments:
            df: dataframe containing the data to analyze
            verbose (Optional): bool, prints the regression summary if True
        """
        results_ols = self.fit_ols(df=df)
        return results_ols.bse[self.treatment_col]

    def analysis_confidence_interval(
        self, df: pd.DataFrame, alpha: float, verbose: bool = False
    ) -> ConfidenceInterval:
        """Returns the confidence interval of the analysis
        Arguments:
            df: dataframe containing the data to analyze
            alpha: significance level
            verbose (Optional): bool, prints the regression summary if True
        """
        results_ols = self.fit_ols(df)
        # Extract the confidence interval for the treatment column
        conf_int_df = results_ols.conf_int(alpha=alpha)
        lower_bound, upper_bound = conf_int_df.loc[self.treatment_col]

        if verbose:
            print(results_ols.summary())

        # Return the confidence interval
        return ConfidenceInterval(lower=lower_bound, upper=upper_bound, alpha=alpha)

    def analysis_inference_results(
        self, df: pd.DataFrame, alpha: float, verbose: bool = False
    ) -> InferenceResults:
        """Returns the inference results of the analysis
        Arguments:
            df: dataframe containing the data to analyze
            alpha: significance level
            verbose (Optional): bool, prints the regression summary if True
        """
        results_ols = self.fit_ols(df)

        std_error = results_ols.bse[self.treatment_col]
        ate = results_ols.params[self.treatment_col]
        p_value = self.pvalue_based_on_hypothesis(results_ols)

        # Extract the confidence interval for the treatment column
        conf_int_df = results_ols.conf_int(alpha=alpha)
        lower_bound, upper_bound = conf_int_df.loc[self.treatment_col]

        if verbose:
            print(results_ols.summary())

        # Return the confidence interval
        return InferenceResults(
            ate=ate,
            p_value=p_value,
            std_error=std_error,
            conf_int=ConfidenceInterval(
                lower=lower_bound, upper=upper_bound, alpha=alpha
            ),
            fitted_model=results_ols,
        )

    @classmethod
    def from_config(cls, config):
        """Creates an OLSAnalysis object from a PowerConfig object"""
        return cls(
            target_col=config.target_col,
            treatment_col=config.treatment_col,
            treatment=config.treatment,
            covariates=config.covariates,
            hypothesis=config.hypothesis,
            cov_type=config.cov_type,
            add_covariate_interaction=config.add_covariate_interaction,
            relative_effect=config.relative_effect,
        )


class ClusteredOLSAnalysis(OLSAnalysis):
    """
    Class to run OLS clustered analysis

    Arguments:
        cluster_cols: list of columns to use as clusters
        target_col: name of the column containing the variable to measure
        treatment_col: name of the column containing the treatment variable
        treatment: name of the treatment to use as the treated group
        covariates: list of columns to use as covariates
        hypothesis: one of "two-sided", "less", "greater" indicating the alternative hypothesis
        add_covariate_interaction: bool, if True, adds interaction terms between covariates and treatment

    Usage:

    ```python
    from cluster_experiments.experiment_analysis import ClusteredOLSAnalysis
    import pandas as pd

    df = pd.DataFrame({
        'x': [1, 2, 3, 0, 0, 1, 2, 0],
        'treatment': ["A"] * 2 + ["B"] * 2 + ["A"] * 2 + ["B"] * 2,
        'cluster': [1, 1, 2, 2, 3, 3, 4, 4],
    })

    ClusteredOLSAnalysis(
        cluster_cols=['cluster'],
        target_col='x',
    ).get_pvalue(df)
    ```
    """

    def __init__(
        self,
        cluster_cols: List[str],
        target_col: str = "target",
        treatment_col: str = "treatment",
        treatment: str = "B",
        covariates: Optional[List[str]] = None,
        hypothesis: str = "two-sided",
        add_covariate_interaction: bool = False,
        relative_effect: bool = False,
    ):
        super().__init__(
            target_col=target_col,
            treatment_col=treatment_col,
            treatment=treatment,
            covariates=covariates,
            hypothesis=hypothesis,
            cov_type="cluster",
            add_covariate_interaction=add_covariate_interaction,
            relative_effect=relative_effect,
        )
        self.cluster_cols = cluster_cols

    def fit_ols(self, df: pd.DataFrame) -> RegressionResultsProtocol:
        """Returns the fitted OLS model"""
        if self.add_covariate_interaction:
            df = self._add_interaction_covariates(df)
        ols_fit = sm.OLS.from_formula(
            self.formula,
            data=df,
        ).fit(
            cov_type=self.cov_type,
            cov_kwds={"groups": self._get_cluster_column(df)},
        )

        # create point estimate, pvalue and std error transformation in case of relative effects
        if self.relative_effect:
            relative_ols_fit = LiftRegressionTransformer(
                treatment_col=self.treatment_col
            )
            relative_ols_fit.fit(
                ols=ols_fit, df=df, covariate_cols=self.covariates_list
            )
            return relative_ols_fit

        return ols_fit

    @classmethod
    def from_config(cls, config):
        """Creates an OLSAnalysis object from a PowerConfig object"""
        return cls(
            target_col=config.target_col,
            treatment_col=config.treatment_col,
            treatment=config.treatment,
            covariates=config.covariates,
            hypothesis=config.hypothesis,
            cluster_cols=config.cluster_cols,
            add_covariate_interaction=config.add_covariate_interaction,
            relative_effect=config.relative_effect,
        )


class MLMExperimentAnalysis(ExperimentAnalysis):
    """
    Class to run Mixed Linear Models clustered analysis

    Arguments:
        cluster_cols: list of columns to use as clusters
        target_col: name of the column containing the variable to measure
        treatment_col: name of the column containing the treatment variable
        treatment: name of the treatment to use as the treated group
        covariates: list of columns to use as covariates
        hypothesis: one of "two-sided", "less", "greater" indicating the alternative hypothesis

    Usage:

    ```python
    from cluster_experiments.experiment_analysis import MLMExperimentAnalysis
    import pandas as pd

    df = pd.DataFrame({
        'x': [1, 2, 3, 0, 0, 1],
        'treatment': ["A"] * 3 + ["B"] * 3,
        'cluster': [1] * 6,
    })

    MLMExperimentAnalysis(
        cluster_cols=['cluster'],
        target_col='x',
    ).get_pvalue(df)
    ```
    """

    def __init__(
        self,
        cluster_cols: List[str],
        target_col: str = "target",
        treatment_col: str = "treatment",
        treatment: str = "B",
        covariates: Optional[List[str]] = None,
        hypothesis: str = "two-sided",
        add_covariate_interaction: bool = False,
    ):
        super().__init__(
            target_col=target_col,
            treatment_col=treatment_col,
            cluster_cols=cluster_cols,
            treatment=treatment,
            covariates=covariates,
            hypothesis=hypothesis,
            add_covariate_interaction=add_covariate_interaction,
        )
        self.re_formula = None
        self.vc_formula = None

    def fit_mlm(self, df: pd.DataFrame) -> sm.MixedLM:
        """Returns the fitted MLM model"""
        if self.add_covariate_interaction:
            df = self._add_interaction_covariates(df)
        return sm.MixedLM.from_formula(
            formula=self.formula,
            data=df,
            groups=self._get_cluster_column(df),
            re_formula=self.re_formula,
            vc_formula=self.vc_formula,
        ).fit()

    def analysis_pvalue(self, df: pd.DataFrame, verbose: bool = False) -> float:
        """Returns the p-value of the analysis
        Arguments:
            df: dataframe containing the data to analyze
            verbose (Optional): bool, prints the regression summary if True
        """
        results_mlm = self.fit_mlm(df)
        if verbose:
            print(results_mlm.summary())

        p_value = self.pvalue_based_on_hypothesis(results_mlm)
        return p_value

    def analysis_point_estimate(self, df: pd.DataFrame, verbose: bool = False) -> float:
        """Returns the point estimate of the analysis
        Arguments:
            df: dataframe containing the data to analyze
            verbose (Optional): bool, prints the regression summary if True
        """
        results_mlm = self.fit_mlm(df)
        return results_mlm.params[self.treatment_col]

    def analysis_standard_error(self, df: pd.DataFrame, verbose: bool = False) -> float:
        """Returns the standard error of the analysis
        Arguments:
            df: dataframe containing the data to analyze
            verbose (Optional): bool, prints the regression summary if True
        """
        results_mlm = self.fit_mlm(df)
        return results_mlm.bse[self.treatment_col]

    def analysis_confidence_interval(
        self, df: pd.DataFrame, alpha: float, verbose: bool = False
    ) -> ConfidenceInterval:
        """Returns the confidence interval of the analysis
        Arguments:
            df: dataframe containing the data to analyze
            alpha: significance level
            verbose (Optional): bool, prints the regression summary if True
        """
        results_mlm = self.fit_mlm(df)
        # Extract the confidence interval for the treatment column
        conf_int_df = results_mlm.conf_int(alpha=alpha)
        lower_bound, upper_bound = conf_int_df.loc[self.treatment_col]

        if verbose:
            print(results_mlm.summary())

        # Return the confidence interval
        return ConfidenceInterval(lower=lower_bound, upper=upper_bound, alpha=alpha)

    def analysis_inference_results(
        self, df: pd.DataFrame, alpha: float, verbose: bool = False
    ) -> InferenceResults:
        """Returns the inference results of the analysis
        Arguments:
            df: dataframe containing the data to analyze
            alpha: significance level
            verbose (Optional): bool, prints the regression summary if True
        """
        results_mlm = self.fit_mlm(df)

        std_error = results_mlm.bse[self.treatment_col]
        ate = results_mlm.params[self.treatment_col]
        p_value = self.pvalue_based_on_hypothesis(results_mlm)

        # Extract the confidence interval for the treatment column
        conf_int_df = results_mlm.conf_int(alpha=alpha)
        lower_bound, upper_bound = conf_int_df.loc[self.treatment_col]

        if verbose:
            print(results_mlm.summary())

        # Return the confidence interval
        return InferenceResults(
            ate=ate,
            p_value=p_value,
            std_error=std_error,
            conf_int=ConfidenceInterval(
                lower=lower_bound, upper=upper_bound, alpha=alpha
            ),
            fitted_model=results_mlm,
        )


class SyntheticControlAnalysis(ExperimentAnalysis):
    """
    Class to run Synthetic control analysis. It expects only one treatment cluster.

    Arguments:

        target_col (str): The name of the column containing the variable to measure.
        treatment_col (str): The name of the column containing the treatment variable.
        treatment (str): The name of the treatment to use as the treated group.
        cluster_cols (list): A list of columns to use as clusters.
        hypothesis (str): One of "two-sided", "less", "greater" indicating the hypothesis.
        time_col (str): The name of the column containing the time data.
        intervention_date (str): The date when the intervention occurred.
    Usage:

    ```python
    from cluster_experiments.experiment_analysis import SyntheticControlAnalysis
    import pandas as pd
    import numpy as np
    from itertools import product

    dates = pd.date_range("2022-01-01", "2022-01-31", freq="d")

    users = [f"User {i}" for i in range(10)]

    # Create a combination of each date with each user
    combinations = list(product(users, dates))

    target_values = np.random.normal(0, 1, size=len(combinations))

    df = pd.DataFrame(combinations, columns=["user", "date"])
    df["target"] = target_values

    df["treatment"] = "A"
    df.loc[(df["user"] == "User 5"), "treatment"] = "B"

    SyntheticControlAnalysis(
        cluster_cols=["user"], time_col="date", intervention_date="2022-01-15"
    ).get_pvalue(df)

    ```
    """

    def __init__(
        self,
        intervention_date: str,
        cluster_cols: List[str],
        target_col: str = "target",
        treatment_col: str = "treatment",
        treatment: str = "B",
        hypothesis: str = "two-sided",
        time_col: str = "date",
    ):
        super().__init__(
            treatment=treatment,
            target_col=target_col,
            treatment_col=treatment_col,
            hypothesis=hypothesis,
            cluster_cols=cluster_cols,
        )

        self.time_col = time_col
        self.intervention_date = intervention_date

        if time_col in cluster_cols:
            raise ValueError("time columns should not be in cluster columns")

    def _fit(self, pre_experiment_df: pd.DataFrame, verbose: bool) -> np.ndarray:
        """Returns the weight of each donor"""

        if not any(pre_experiment_df[self.treatment_col] == 1):
            raise ValueError("No treatment unit found in the data.")

        X = (
            pre_experiment_df.query(f"{self.treatment_col} == 0")
            .pivot(index=self.cluster_cols, columns=self.time_col)[self.target_col]
            .T
        )

        y = (
            pre_experiment_df.query(f"{self.treatment_col} == 1")
            .pivot(index=self.cluster_cols, columns=self.time_col)[self.target_col]
            .T.iloc[:, 0]
        )

        weights = get_w(X, y, verbose)

        return weights

    def _predict(
        self, df: pd.DataFrame, weights: np.ndarray, treatment_cluster: str
    ) -> pd.DataFrame:
        """
        This method adds a column with the synthetic results and filter only the treatment unit.

        First, it calculates the weights of each donor in the control group using the `fit_synthetic` method.
        It then uses these weights to create a synthetic control group that closely matches the treatment unit before the intervention.
        The synthetic control group is added to the treatment unit in the dataframe.
        """
        synthetic = (
            df[self._get_cluster_column(df) != treatment_cluster]
            .pivot(index=self.time_col, columns=self.cluster_cols)[self.target_col]
            .values.dot(weights)
        )

        # add synthetic to treatment cluster
        return df[self._get_cluster_column(df) == treatment_cluster].assign(
            synthetic=synthetic
        )

    def fit_predict_synthetic(
        self,
        df: pd.DataFrame,
        pre_experiment_df: pd.DataFrame,
        treatment_cluster: str,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Fit the synthetic control model and predict the results for the treatment cluster.
        Args:
            df: The dataframe containing the data after the intervention.
            pre_experiment_df: The dataframe containing the data before the intervention.
            treatment_cluster: The name of the treatment cluster.
            verbose: If True, print the status of the optimization of weights.

        Returns:
            The dataframe with the synthetic results added to the treatment cluster.
        """
        weights = self._fit(pre_experiment_df=pre_experiment_df, verbose=verbose)

        prediction = self._predict(
            df=df, weights=weights, treatment_cluster=treatment_cluster
        )
        return prediction

    def pvalue_based_on_hypothesis(
        self, ate: np.float64, avg_effects: Dict[str, float]
    ) -> float:
        """
        Returns the p-value of the analysis.
        1. Count how many times the average effect is greater than the real treatment unit
        2. Average it with the number of units. The result is the p-value using Fisher permutation exact test.
        """

        avg_effects = list(avg_effects.values())

        if HypothesisEntries(self.hypothesis) == HypothesisEntries.LESS:
            return np.mean(avg_effects < ate)
        if HypothesisEntries(self.hypothesis) == HypothesisEntries.GREATER:
            return np.mean(avg_effects > ate)
        if HypothesisEntries(self.hypothesis) == HypothesisEntries.TWO_SIDED:
            avg_effects = np.abs(avg_effects)
            return np.mean(avg_effects > ate)

        raise ValueError(f"{self.hypothesis} is not a valid HypothesisEntries")

    def _get_treatment_cluster(self, df: pd.DataFrame) -> str:
        """Returns the first treatment cluster. The current implementation of Synthetic Control only accepts one treatment cluster.
        This will be left inside Synthetic class because it doesn't apply for other analyses
        """
        treatment_df = df[df[self.treatment_col] == 1]
        treatment_cluster = self._get_cluster_column(treatment_df).unique()[0]
        return treatment_cluster

    def analysis_pvalue(self, df: pd.DataFrame, verbose: bool = False) -> float:
        """
        Returns the p-value of the analysis.
        1. Calculate the average effect after intervention for each unit.
        2. Count how many times the average effect is greater than the real treatment unit
        3. Average it with the number of units. The result is the p-value using Fisher permutation test
        """

        clusters = self._get_cluster_column(df).unique()
        treatment_cluster = self._get_treatment_cluster(df)

        synthetic_donors = {
            cluster: self.analysis_point_estimate(
                treatment_cluster=cluster,
                df=df,
                verbose=verbose,
            )
            for cluster in clusters
        }

        ate = synthetic_donors[treatment_cluster]
        synthetic_donors.pop(treatment_cluster)

        return self.pvalue_based_on_hypothesis(ate=ate, avg_effects=synthetic_donors)

    def analysis_point_estimate(
        self,
        df: pd.DataFrame,
        treatment_cluster: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Calculate the point estimate for the treatment effect for a specified cluster by averaging across the time windows.
        """
        df, pre_experiment_df = self._split_pre_experiment_df(df)

        if treatment_cluster is None:
            treatment_cluster = self._get_treatment_cluster(df)

        df = self.fit_predict_synthetic(
            df, pre_experiment_df, treatment_cluster, verbose=verbose
        )

        df["effect"] = df[self.target_col] - df["synthetic"]
        avg_effect = df["effect"].mean()
        return avg_effect

    def _split_pre_experiment_df(self, df: pd.DataFrame):
        """Split the dataframe into pre-experiment and experiment dataframes"""
        pre_experiment_df = df[(df[self.time_col] <= self.intervention_date)]
        df = df[(df[self.time_col] > self.intervention_date)]
        return df, pre_experiment_df


class DeltaMethodAnalysis(ExperimentAnalysis):
    n_clusters_warning_limit = 1000

    def __init__(
        self,
        cluster_cols: List[str],
        target_col: str = "target",
        scale_col: str = "scale",
        treatment_col: str = "treatment",
        treatment: str = "B",
        covariates: Optional[List[str]] = None,
        hypothesis: str = "two-sided",
    ):
        """
        Class to run the Delta Method approximation for estimating the treatment effect on a ratio metric (target/scale) under a clustered design.
        The analysis is done on the aggregated data at the cluster level, making computation more efficient.

        Arguments:
            cluster_cols: list of columns to use as clusters.
            target_col: name of the column containing the variable to measure (the numerator of the ratio).
            scale_col: name of the column containing the scale variable (the denominator of the ratio).
            treatment_col: name of the column containing the treatment variable.
            treatment: name of the treatment to use as the treated group.
            covariates: list of columns to use as covariates. Have to be previously aggregated at the cluster level.
            hypothesis: one of "two-sided", "less", "greater" indicating the alternative hypothesis.

        Usage:
        ```python
        import pandas as pd

        from cluster_experiments.experiment_analysis import DeltaMethodAnalysis

        df = pd.DataFrame({
            'x': [1, 2, 3, 0, 0, 1] * 2,
            'y': [2, 2, 5, 1, 1, 1] * 2,
            'treatment': ["A"] * 6 + ["B"] * 6,
            'cluster': [1, 2, 3, 1, 2, 3] * 2,
            'z': [1, 2, 3, 4, 5, 6] * 2,
        })

        DeltaMethodAnalysis(
            cluster_cols=['cluster'],
            target_col='x',
            scale_col='y',
        ).get_pvalue(df)
        ```
        """

        super().__init__(
            target_col=target_col,
            treatment_col=treatment_col,
            cluster_cols=cluster_cols,
            treatment=treatment,
            covariates=covariates,
            hypothesis=hypothesis,
        )
        self.scale_col = scale_col
        self.cluster_cols = cluster_cols or []
        self.covariates = covariates or []

    def _compute_thetas(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Computes the theta value for the CUPED method.
        Thetas are computed as the inverse of the covariance matrix of covariates multiplied by the covariance between covariates and target metric.

        Delta method with CUPED should work like the following. For each randomization unit i we observe $Y_i$, $N_i$.

        Refer to delta.md for mathematical details.
        """
        target = np.asarray(df[self.target_col])
        scale = np.asarray(df[self.scale_col])
        covariates = np.asarray(df[self.covariates])

        if len(self.covariates) == 1:
            # Code for n = 1 (deng et al)
            Y, N = target, scale

            # need to covariate * scale to get X_i = covariate_i * N_i
            X, M = np.squeeze(covariates) * scale, scale
            sigma = np.cov([Y, N, X, M])  # 4

            mu_Y, mu_N = Y.mean(), N.mean()
            mu_X, mu_M = X.mean(), M.mean()
            beta1 = np.array([1 / mu_N, -mu_Y / mu_N**2, 0, 0]).T
            beta2 = np.array([0, 0, 1 / mu_M, -mu_X / mu_M**2]).T

            # formula from Deng et al. n's would cancel out
            theta = np.dot(beta1, np.matmul(sigma, beta2)) / np.dot(
                beta2, np.matmul(sigma, beta2)
            )
            return {"theta": np.array([theta])}

        # Sample means
        Y, N, M = target, scale, scale
        X = covariates * scale.reshape(-1, 1)
        sigma = np.cov(np.column_stack([Y, N, X, M]), rowvar=False, ddof=0)

        mu_Y, mu_N = Y.mean(), N.mean()
        mu_X, mu_M = X.mean(axis=0), M.mean()
        k = len(self.covariates)

        # numerator (follow delta.md)
        cov_YX = sigma[0, 2 : 2 + k]
        cov_NX = sigma[1, 2 : 2 + k]
        cov_YN = sigma[0, 1]
        cov_NN = sigma[1, 1]
        numerator = (
            cov_YX / mu_N**2
            - (mu_Y / mu_N**2) * (cov_NX / mu_N)
            - mu_X * (cov_YN / mu_N**3)
            + mu_X * mu_Y * (cov_NN / mu_N**4)
        )

        # denominator (follow delta.md, K * D * K^T)
        d_matrix = sigma[2:, 2:]  # (k + 1) x (k + 1)
        k_matrix = np.zeros((k, k + 1))
        k_matrix[np.diag_indices(k)] = 1 / mu_M
        k_matrix[:, -1] = -mu_X / (mu_M**2)
        denominator = k_matrix @ d_matrix @ k_matrix.T

        theta = np.dot(np.linalg.pinv(denominator), numerator)
        # return both theta and pieces for variance calculation
        return {"theta": theta, "numerator": numerator, "denominator": denominator}

    def _get_ratio_variance_simple(self, df: pd.DataFrame) -> float:
        """
        Variance of the ratio of means (sum(target)/sum(scale)) via delta method.

        Parameters
        ----------
        target : array-like
            Numerator per unit.
        scale : array-like
            Denominator per unit.

        Returns
        -------
        variance : float
            Estimated variance of the ratio metric.
        """
        target = np.asarray(df[self.target_col])
        scale = np.asarray(df[self.scale_col])

        target_mean, scale_mean = np.mean(target), np.mean(scale)

        # Sample variances and covariance
        var_target, var_scale = np.var(target, ddof=0), np.var(scale, ddof=0)
        cov_target_scale = np.cov(target, scale, ddof=0)[0, 1]

        # Gradient of g(Ȳ, scalē) = Ȳ / scalē
        grad_target = 1.0 / scale_mean
        grad_scale = -target_mean / (scale_mean**2)

        # Delta method variance
        var_ratio = (
            (grad_target**2) * var_target
            + (grad_scale**2) * var_scale
            + 2 * grad_target * grad_scale * cov_target_scale
        )

        return var_ratio / len(target)

    def _get_ratio_variance_cuped(
        self, df: pd.DataFrame, thetas: Optional[Dict[str, np.ndarray]]
    ) -> float:
        """
        Y-only CUPED delta variance for the ratio of means on this group's rows.
        Uses pooled theta, but per-arm moments for the variance pieces.
        """
        # data
        target = df[self.target_col].to_numpy()
        scale = df[self.scale_col].to_numpy()
        covariates = np.asarray(df[self.covariates])

        if len(self.covariates) == 1:
            # Code for n = 1 (deng et al)
            Y, N = target, scale
            X, M = np.squeeze(covariates) * scale, scale
            sigma = np.cov([Y, N, X, M])  # 4

            mu_Y, mu_N = Y.mean(), N.mean()
            # M is the same as N, in general it could be different,
            # but we're copying Deng et al. here and it's easier
            mu_X, mu_M = X.mean(), M.mean()
            # the betas are from the deng paper
            beta1 = np.array([1 / mu_N, -mu_Y / mu_N**2, 0, 0]).T
            beta2 = np.array([0, 0, 1 / mu_M, -mu_X / mu_M**2]).T

            var_Y_div_N = np.dot(beta1, np.matmul(sigma, beta1.T))
            var_X_div_M = np.dot(beta2, np.matmul(sigma, beta2))
            cov = np.dot(beta1, np.matmul(sigma, beta2))

            # theta is a scalar in this case
            theta = thetas["theta"][0]

            # can also use traditional delta method for the var_Y_div_N type terms
            return (var_Y_div_N + (theta**2) * var_X_div_M - 2 * theta * cov) / len(
                target
            )

        # multiple covariates
        variance_simple = self._get_ratio_variance_simple(df) * len(target)
        theta = thetas["theta"]
        numerator = thetas["numerator"]
        denominator = thetas["denominator"]

        # follow delta.md notation, but in general it's var(Y/N) + theta Var(X/M) theta^T - 2 theta cov(Y/N, X/M)
        var_cuped = (
            variance_simple + (theta @ denominator @ theta) - 2 * (theta @ numerator)
        )
        return var_cuped / len(target)

    def _aggregate_to_cluster(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns an aggregated dataframe of the target and scale variables at the cluster (and treatment) level.

        Arguments:
            df: dataframe containing the data to analyze
        """
        group_cols = self.cluster_cols + [self.treatment_col]
        aggregate_df = df.groupby(by=group_cols, as_index=False).agg(
            {self.target_col: "sum", self.scale_col: "sum"}
        )
        return aggregate_df

    def _correct_target(
        self,
        df: pd.DataFrame,
        thetas: Optional[Dict[str, np.ndarray]],
        covariates_means: List[float],
    ) -> pd.Series:
        """
        Corrects the target variable using thetas and covariates means.
        If thetas are not provided, it returns the original target.
        """
        if len(self.covariates) == 0 or thetas is None:
            return df[self.target_col]

        # Apply correction using thetas and covariates means
        corrected_target = df[self.target_col].copy()
        for covariate, theta, mean in zip(
            self.covariates, thetas["theta"], covariates_means
        ):
            # According to deng, covariate is supposed to be at the lower granularity level
            # so it predicts the ratio target/scale
            corrected_target -= theta * (df[covariate] - mean) * df[self.scale_col]
        return corrected_target

    def _get_ratio_variance(
        self, df: pd.DataFrame, thetas: Optional[Dict[str, np.ndarray]]
    ) -> float:
        """
        Returns the variance of the ratio metric (target/scale) as estimated by the delta method.
        If covariates are given, variance reduction is used.
        """
        if self.covariates:
            return self._get_ratio_variance_cuped(df, thetas)
        else:
            return self._get_ratio_variance_simple(df)

    def _get_group_mean_and_variance(
        self,
        df: pd.DataFrame,
        thetas: Optional[Dict[str, np.ndarray]],
        covariates_means: List[float],
    ) -> tuple[float, float]:
        """
        Returns the mean and variance of the ratio metric (target/scale) as estimated by the delta method for a given group (treatment).
        If covariates are given, variance reduction is used. For it to work, the dataframe must be aggregated first at the cluster level, so no assumptions on aggregation of covariates has to be done.

        Arguments:
            df: dataframe containing the data to analyze.
        """
        corrected_target = self._correct_target(df, thetas, covariates_means)
        group_mean = sum(corrected_target) / sum(df[self.scale_col])

        if self.covariates:
            group_variance = self._get_ratio_variance(df, thetas)
        else:
            group_variance = self._get_ratio_variance_simple(df)

        # Return the mean and variance of the ratio metric
        return group_mean, group_variance

    def _get_mean_standard_error(self, df: pd.DataFrame) -> tuple[float, float]:
        """
        Returns mean and variance of the ratio metric (target/scale) for a given cluster (i.e. user) computed using the Delta Method.
        Variance reduction is used if covariates are given.
        """

        if (self._get_num_clusters(df) < self.n_clusters_warning_limit).any():
            self.__warn_small_group_size()

        if self.covariates:
            self.__check_data_is_aggregated(df)
        else:
            df = self._aggregate_to_cluster(df)

        is_treatment = df[self.treatment_col] == 1

        thetas_dict = self._compute_thetas(df) if self.covariates else None
        covariates_means = [
            df[covariate].sum() / df[self.scale_col].sum()
            for covariate in self.covariates
        ]

        treat_mean, treat_var = self._get_group_mean_and_variance(
            df[is_treatment], thetas_dict, covariates_means
        )
        ctrl_mean, ctrl_var = self._get_group_mean_and_variance(
            df[~is_treatment], thetas_dict, covariates_means
        )

        mean_diff = treat_mean - ctrl_mean
        standard_error = np.sqrt(treat_var + ctrl_var)

        return mean_diff, standard_error

    def analysis_pvalue(self, df: pd.DataFrame) -> float:
        """
        Returns the p-value of the analysis.

        Arguments:
            df: dataframe containing the data to analyze.
        """

        mean_diff, standard_error = self._get_mean_standard_error(df)

        z_score = mean_diff / standard_error
        p_value = 2 * (1 - norm.cdf(abs(z_score)))

        results_delta = ModelResults(
            params={self.treatment_col: mean_diff},
            pvalues={self.treatment_col: p_value},
        )

        p_value = self.pvalue_based_on_hypothesis(results_delta)

        return p_value

    def analysis_point_estimate(self, df: pd.DataFrame) -> float:
        """Returns the point estimate of the analysis
        Arguments:
            df: dataframe containing the data to analyze
            verbose (Optional): bool, prints the regression summary if True
        """
        mean_diff, _standard_error = self._get_mean_standard_error(df)
        return mean_diff

    def analysis_standard_error(self, df: pd.DataFrame) -> float:
        """Returns the standard error of the analysis
        Arguments:
            df: dataframe containing the data to analyze
            verbose (Optional): bool, prints the regression summary if True
        """
        _mean_diff, standard_error = self._get_mean_standard_error(df)
        return standard_error

    def analysis_confidence_interval(
        self, df: pd.DataFrame, alpha: float, verbose: bool = False
    ) -> ConfidenceInterval:
        """Returns the confidence interval of the analysis
        Arguments:
            df: dataframe containing the data to analyze
            alpha: significance level
        """
        ate, std_error = self._get_mean_standard_error(df)

        z_score = ate / std_error
        p_value = 2 * (1 - norm.cdf(abs(z_score)))

        results_delta = ModelResults(
            params={self.treatment_col: ate},
            pvalues={self.treatment_col: p_value},
        )

        p_value = self.pvalue_based_on_hypothesis(results_delta)

        # Extract the confidence interval for the treatment column
        crit_z_score = norm.ppf(1 - alpha / 2)
        conf_int = crit_z_score * std_error
        lower_bound, upper_bound = ate - conf_int, ate + conf_int

        # Return the confidence interval
        return ConfidenceInterval(lower=lower_bound, upper=upper_bound, alpha=alpha)

    def analysis_inference_results(
        self, df: pd.DataFrame, alpha: float, verbose: bool = False
    ) -> InferenceResults:
        """Returns the inference results of the analysis
        Arguments:
            df: dataframe containing the data to analyze
            alpha: significance level
        """
        ate, std_error = self._get_mean_standard_error(df)

        z_score = ate / std_error
        p_value = 2 * (1 - norm.cdf(abs(z_score)))

        results_delta = ModelResults(
            params={self.treatment_col: ate},
            pvalues={self.treatment_col: p_value},
        )

        p_value = self.pvalue_based_on_hypothesis(results_delta)

        # Extract the confidence interval for the treatment column
        crit_z_score = norm.ppf(1 - alpha / 2)
        conf_int = crit_z_score * std_error
        lower_bound, upper_bound = ate - conf_int, ate + conf_int

        # Return the confidence interval
        return InferenceResults(
            ate=ate,
            p_value=p_value,
            std_error=std_error,
            conf_int=ConfidenceInterval(
                lower=lower_bound, upper=upper_bound, alpha=alpha
            ),
        )

    @classmethod
    def from_config(cls, config):
        """Creates a DeltaMethodAnalysis object from a PowerConfig object"""
        return cls(
            cluster_cols=config.cluster_cols,
            target_col=config.target_col,
            scale_col=config.scale_col,
            treatment_col=config.treatment_col,
            treatment=config.treatment,
            hypothesis=config.hypothesis,
            covariates=config.covariates,
        )

    def __check_data_is_aggregated(self, df):
        """
        Check if the data is already aggregated at the cluster level.
        """

        if df.groupby(self.cluster_cols).size().max() > 1:
            raise ValueError(
                "The data should be aggregated at the cluster level for the Delta Method analysis using covariates."
            )

    def _get_num_clusters(self, df):
        """
        Check if there are enough clusters to run the analysis.
        """
        return df.groupby(self.treatment_col).apply(
            lambda x: self._get_cluster_column(x).nunique()
        )

    def __warn_small_group_size(self):
        warnings.warn(
            "Delta Method approximation may not be accurate for small group sizes"
        )
