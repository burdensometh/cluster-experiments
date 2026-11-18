from dataclasses import asdict, dataclass, field
from typing import List

import pandas as pd


@dataclass
class AnalysisPlanResults:
    """
    A dataclass used to represent the results of the experiment analysis.

    Returned by :meth:`AnalysisPlan.analyze <cluster_experiments.inference.analysis_plan.AnalysisPlan.analyze>`.
    Each row corresponds to one hypothesis test (metric × variant × dimension slice).
    Only aggregate fields are stored (ATE, p-value, etc.); for the full underlying
    model fit (e.g. statsmodels regression table) for a single run, use
    :meth:`ExperimentAnalysis.get_inference_results
    <cluster_experiments.experiment_analysis.ExperimentAnalysis.get_inference_results>`
    and then :meth:`InferenceResults.model_summary
    <cluster_experiments.experiment_analysis.InferenceResults.model_summary>`.

    Attributes
    ----------
    metric_alias : List[str]
        The alias of the metric used in the test
    control_variant_name : List[str]
        The name of the control variant
    treatment_variant_name : List[str]
        The name of the treatment variant
    control_variant_mean : List[float]
        The mean value of the control variant
    treatment_variant_mean : List[float]
        The mean value of the treatment variant
    analysis_type : List[str]
        The type of analysis performed
    ate : List[float]
        The average treatment effect
    ate_ci_lower : List[float]
        The lower bound of the confidence interval for the ATE
    ate_ci_upper : List[float]
        The upper bound of the confidence interval for the ATE
    p_value : List[float]
        The p-value of the test
    std_error : List[float]
        The standard error of the test
    dimension_name : List[str]
        The name of the dimension
    dimension_value : List[str]
        The value of the dimension
    alpha: List[float]
        The significance level of the test
    """

    metric_alias: List[str] = field(default_factory=lambda: [])
    control_variant_name: List[str] = field(default_factory=lambda: [])
    treatment_variant_name: List[str] = field(default_factory=lambda: [])
    control_variant_mean: List[float] = field(default_factory=lambda: [])
    treatment_variant_mean: List[float] = field(default_factory=lambda: [])
    analysis_type: List[str] = field(default_factory=lambda: [])
    ate: List[float] = field(default_factory=lambda: [])
    ate_ci_lower: List[float] = field(default_factory=lambda: [])
    ate_ci_upper: List[float] = field(default_factory=lambda: [])
    p_value: List[float] = field(default_factory=lambda: [])
    std_error: List[float] = field(default_factory=lambda: [])
    dimension_name: List[str] = field(default_factory=lambda: [])
    dimension_value: List[str] = field(default_factory=lambda: [])
    alpha: List[float] = field(default_factory=lambda: [])

    def __add__(self, other):
        if not isinstance(other, AnalysisPlanResults):
            return NotImplemented

        return AnalysisPlanResults(
            metric_alias=self.metric_alias + other.metric_alias,
            control_variant_name=self.control_variant_name + other.control_variant_name,
            treatment_variant_name=self.treatment_variant_name
            + other.treatment_variant_name,
            control_variant_mean=self.control_variant_mean + other.control_variant_mean,
            treatment_variant_mean=self.treatment_variant_mean
            + other.treatment_variant_mean,
            analysis_type=self.analysis_type + other.analysis_type,
            ate=self.ate + other.ate,
            ate_ci_lower=self.ate_ci_lower + other.ate_ci_lower,
            ate_ci_upper=self.ate_ci_upper + other.ate_ci_upper,
            p_value=self.p_value + other.p_value,
            std_error=self.std_error + other.std_error,
            dimension_name=self.dimension_name + other.dimension_name,
            dimension_value=self.dimension_value + other.dimension_value,
            alpha=self.alpha + other.alpha,
        )

    def to_dataframe(self):
        return pd.DataFrame(asdict(self))

    def __str__(self) -> str:
        n = len(self.ate)
        if n == 0:
            return "AnalysisPlanResults(0 tests)"
        ate_range = f"[{min(self.ate):.4g}, {max(self.ate):.4g}]"
        return f"AnalysisPlanResults({n} test(s): metrics={self.metric_alias}, ATE in {ate_range})"

    def summary(self) -> str:
        """Return a summary of the analysis plan results."""
        df = self.to_dataframe()
        if df.empty:
            return "Analysis plan results: no tests."
        lines = [
            "Analysis plan results",
            f"  Number of tests: {len(df)}",
            "",
            df.to_string(index=False),
        ]
        return "\n".join(lines)


@dataclass
class AnalysisPlanRandomizationInferenceResult:
    """One randomization-inference test produced by
    :meth:`AnalysisPlan.randomization_inference
    <cluster_experiments.inference.analysis_plan.AnalysisPlan.randomization_inference>`.

    Each row identifies the hypothesis test (metric, variants, dimension slice)
    and stores the observed statistic, the finite-randomization p-value and the
    details needed to audit the rule that produced it.

    Attributes
    ----------
    metric_alias : str
        The alias of the metric used in the test.
    control_variant_name : str
        The name of the control variant.
    treatment_variant_name : str
        The name of the treatment variant.
    dimension_name : str
        The name of the dimension.
    dimension_value : str
        The value of the dimension.
    analysis_type : str
        The type of analysis fitted on each re-assignment.
    observed_statistic : float
        Statistic computed on the observed data.
    p_value : float
        Finite-randomization p-value.
    n_randomizations : int
        Number of valid legal re-assignments ``R`` used in the denominator.
    n_extreme : int
        Number of null statistics at least as extreme (ties included).
    n_ties : int
        Number of null statistics exactly equal to the observed one.
    hypothesis : str
        One of ``"two-sided"``, ``"less"``, ``"greater"``.
    alpha : float
        Significance level stored for reporting purposes.
    """

    metric_alias: str
    control_variant_name: str
    treatment_variant_name: str
    dimension_name: str
    dimension_value: str
    analysis_type: str
    observed_statistic: float
    p_value: float
    n_randomizations: int
    n_extreme: int
    n_ties: int
    hypothesis: str
    alpha: float

    @property
    def ate(self) -> float:
        """Alias for the observed statistic."""
        return self.observed_statistic

    @property
    def p_value_rule(self) -> str:
        """Human-readable statement of the finite-randomization p-value rule."""
        tail = {
            "greater": "null statistics >= observed",
            "less": "null statistics <= observed",
            "two-sided": "|null statistics| >= |observed|",
        }[self.hypothesis]
        return (
            f"p = (1 + #{{{tail}, ties included}}) / (1 + R) with "
            f"R = {self.n_randomizations} finite randomizations"
        )

    def summary(self) -> str:
        """Return a summary of this randomization-inference test."""
        return "\n".join(
            [
                "Randomization inference result",
                f"  Metric: {self.metric_alias}",
                f"  Control: {self.control_variant_name}",
                f"  Treatment: {self.treatment_variant_name}",
                f"  Dimension: {self.dimension_name}={self.dimension_value}",
                f"  Analysis: {self.analysis_type}",
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
class AnalysisPlanRandomizationInferenceResults:
    """Results of :meth:`AnalysisPlan.randomization_inference`.

    Holds one :class:`AnalysisPlanRandomizationInferenceResult` per hypothesis test and
    exposes the same tabular helpers as :class:`AnalysisPlanResults`.

    Attributes
    ----------
    results : List[AnalysisPlanRandomizationInferenceResult]
        One entry per metric × variant × dimension slice.
    """

    results: List[AnalysisPlanRandomizationInferenceResult] = field(
        default_factory=lambda: []
    )

    def __add__(
        self, other: "AnalysisPlanRandomizationInferenceResults"
    ) -> "AnalysisPlanRandomizationInferenceResults":
        if not isinstance(other, AnalysisPlanRandomizationInferenceResults):
            return NotImplemented
        return AnalysisPlanRandomizationInferenceResults(
            results=self.results + other.results
        )

    def __len__(self) -> int:
        return len(self.results)

    def __str__(self) -> str:
        n = len(self.results)
        if n == 0:
            return "AnalysisPlanRandomizationInferenceResults(0 tests)"
        p_values = [r.p_value for r in self.results]
        return (
            f"AnalysisPlanRandomizationInferenceResults({n} test(s): "
            f"metrics={[r.metric_alias for r in self.results]}, "
            f"p-values in [{min(p_values):.4g}, {max(p_values):.4g}])"
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return one row per test with the randomization-inference columns."""
        return pd.DataFrame([asdict(result) for result in self.results])

    def summary(self) -> str:
        """Return a summary of the randomization-inference results."""
        if not self.results:
            return "Analysis plan randomization inference results: no tests."
        return "\n".join(
            [
                "Analysis plan randomization inference results",
                f"  Number of tests: {len(self.results)}",
                "",
                self.to_dataframe().to_string(index=False),
            ]
        )
