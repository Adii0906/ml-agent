"""Registry of the algorithms ML-Agent actually trains.

Every algorithm advertised to the UI is constructed here. Hyperparameters are
validated against an explicit schema; unknown algorithms or parameters raise
``AlgorithmError`` — there is no silent fallback to another model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:  # optional dependency
    from xgboost import XGBClassifier, XGBRegressor

    XGBOOST_AVAILABLE = True
except Exception:  # pragma: no cover - depends on environment
    XGBClassifier = XGBRegressor = None
    XGBOOST_AVAILABLE = False


class AlgorithmError(ValueError):
    pass


@dataclass
class Param:
    name: str
    label: str
    type: str  # int | float | choice | bool | int_or_none
    default: Any
    help: str
    min: Optional[float] = None
    max: Optional[float] = None
    choices: Optional[List[Any]] = None
    log: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if v is not None}
        return d

    def coerce(self, value: Any) -> Any:
        if self.type == "bool":
            if isinstance(value, bool):
                return value
            if isinstance(value, str) and value.lower() in ("true", "false"):
                return value.lower() == "true"
            raise AlgorithmError(f"'{self.label}' must be true or false.")
        if self.type == "choice":
            if value not in (self.choices or []):
                raise AlgorithmError(f"'{self.label}' must be one of {self.choices}.")
            return value
        if self.type == "int_or_none" and (value is None or value == ""):
            return None
        try:
            num = float(value)
        except (TypeError, ValueError):
            raise AlgorithmError(f"'{self.label}' must be a number.") from None
        if self.type in ("int", "int_or_none"):
            if num != int(num):
                raise AlgorithmError(f"'{self.label}' must be a whole number.")
            num = int(num)
        if self.min is not None and num < self.min:
            raise AlgorithmError(f"'{self.label}' must be ≥ {self.min}.")
        if self.max is not None and num > self.max:
            raise AlgorithmError(f"'{self.label}' must be ≤ {self.max}.")
        return num


@dataclass
class Algorithm:
    id: str
    name: str
    task: str  # classification | regression
    family: str
    summary: str
    strengths: List[str]
    limitations: List[str]
    scale_sensitive: bool
    builder: Callable[[Dict[str, Any], int], Any]
    params: List[Param] = field(default_factory=list)
    available: bool = True
    unavailable_reason: Optional[str] = None
    aliases: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "task": self.task, "family": self.family,
            "summary": self.summary, "strengths": self.strengths, "limitations": self.limitations,
            "scale_sensitive": self.scale_sensitive,
            "params": [p.to_dict() for p in self.params],
            "available": self.available, "unavailable_reason": self.unavailable_reason,
        }

    def validate_params(self, raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        raw = dict(raw or {})
        known = {p.name: p for p in self.params}
        unknown = sorted(set(raw) - set(known))
        if unknown:
            raise AlgorithmError(f"{self.name} does not accept: {', '.join(unknown)}.")
        resolved = {}
        for p in self.params:
            resolved[p.name] = p.coerce(raw[p.name]) if p.name in raw else p.default
        return resolved

    def build(self, params: Dict[str, Any], random_state: int):
        if not self.available:
            raise AlgorithmError(f"{self.name} is unavailable: {self.unavailable_reason}")
        return self.builder(params, random_state)


# ----------------------------------------------------------- shared params
def _class_weight() -> Param:
    return Param("class_weight", "Class weight", "choice", "none",
                 "'balanced' re-weights classes inversely to their frequency — useful for imbalanced targets.",
                 choices=["none", "balanced"])


def _cw(v: str):
    return None if v == "none" else v


def _n_estimators(default=200) -> Param:
    return Param("n_estimators", "Trees", "int", default, "Number of trees. More trees reduce variance but cost time.", 10, 2000)


def _max_depth(default=None) -> Param:
    return Param("max_depth", "Max depth", "int_or_none", default,
                 "Maximum tree depth. Empty means unlimited (risk of overfitting).", 1, 100)


def _min_samples_leaf() -> Param:
    return Param("min_samples_leaf", "Min samples per leaf", "int", 1, "Larger values smooth the model.", 1, 1000)


ALGORITHMS: List[Algorithm] = [
    # ---------------------------------------------------- classification
    Algorithm(
        "logistic_regression", "Logistic Regression", "classification", "Linear",
        "A linear model of class log-odds. The standard, interpretable baseline.",
        ["Fast and stable", "Coefficients are interpretable", "Well-calibrated probabilities"],
        ["Only linear decision boundaries", "Needs scaled features", "Sensitive to strongly correlated features"],
        True,
        lambda p, rs: LogisticRegression(C=p["C"], class_weight=_cw(p["class_weight"]), max_iter=p["max_iter"], random_state=rs),
        [Param("C", "Inverse regularisation (C)", "float", 1.0, "Smaller values mean stronger regularisation.", 1e-4, 1e4, log=True),
         Param("max_iter", "Max iterations", "int", 1000, "Solver iteration budget.", 100, 20000),
         _class_weight()],
        aliases=["logistic", "logistic regression"],
    ),
    Algorithm(
        "decision_tree_classifier", "Decision Tree", "classification", "Tree",
        "A single tree of if/else splits on feature thresholds.",
        ["Handles non-linear relationships", "No scaling needed", "Easy to visualise and explain"],
        ["Overfits easily without depth limits", "High variance between runs on different data"],
        False,
        lambda p, rs: DecisionTreeClassifier(max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"],
                                             class_weight=_cw(p["class_weight"]), random_state=rs),
        [_max_depth(8), _min_samples_leaf(), _class_weight()],
        aliases=["decision tree", "decision tree classifier"],
    ),
    Algorithm(
        "random_forest_classifier", "Random Forest", "classification", "Ensemble",
        "Many decorrelated decision trees whose votes are averaged.",
        ["Strong default performance on tabular data", "Robust to outliers and feature scale", "Built-in feature importance"],
        ["Larger and slower than a single tree", "Less interpretable", "Importances biased toward high-cardinality features"],
        False,
        lambda p, rs: RandomForestClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                                             min_samples_leaf=p["min_samples_leaf"], class_weight=_cw(p["class_weight"]),
                                             n_jobs=-1, random_state=rs),
        [_n_estimators(), _max_depth(), _min_samples_leaf(), _class_weight()],
        aliases=["random forest", "random forest classifier", "rf"],
    ),
    Algorithm(
        "gradient_boosting_classifier", "Gradient Boosting", "classification", "Ensemble",
        "Trees added sequentially, each correcting the errors of the previous ones.",
        ["Often the most accurate on small/medium tabular data", "Captures interactions"],
        ["Sequential, so slower to train", "More hyperparameters to tune", "Can overfit with high learning rates"],
        False,
        lambda p, rs: GradientBoostingClassifier(n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
                                                 max_depth=p["max_depth"], random_state=rs),
        [_n_estimators(150), Param("learning_rate", "Learning rate", "float", 0.1, "Shrinks each tree's contribution.", 0.001, 1.0, log=True),
         Param("max_depth", "Max depth", "int", 3, "Depth of each tree.", 1, 12)],
        aliases=["gradient boosting", "gbm"],
    ),
    Algorithm(
        "hist_gradient_boosting_classifier", "Histogram Gradient Boosting", "classification", "Ensemble",
        "Gradient boosting on binned features — scikit-learn's fast boosting for larger datasets.",
        ["Scales to hundreds of thousands of rows", "Strong accuracy", "Early stopping on large data"],
        ["Less interpretable", "Overkill for tiny datasets"],
        False,
        lambda p, rs: HistGradientBoostingClassifier(max_iter=p["max_iter"], learning_rate=p["learning_rate"],
                                                     max_leaf_nodes=p["max_leaf_nodes"], class_weight=_cw(p["class_weight"]), random_state=rs),
        [Param("max_iter", "Boosting iterations", "int", 200, "Number of boosting rounds.", 10, 5000),
         Param("learning_rate", "Learning rate", "float", 0.1, "Shrinks each tree's contribution.", 0.001, 1.0, log=True),
         Param("max_leaf_nodes", "Max leaf nodes", "int", 31, "Leaves per tree.", 2, 512),
         _class_weight()],
        aliases=["hist gradient boosting", "histgradientboosting"],
    ),
    Algorithm(
        "svc", "Support Vector Machine", "classification", "Kernel",
        "Finds the maximum-margin boundary, optionally in a kernel-transformed space.",
        ["Effective with many features", "Flexible non-linear boundaries via kernels"],
        ["Training time grows roughly quadratically with rows", "Needs scaled features", "No native probabilities (decision scores are used for ROC-AUC)"],
        True,
        lambda p, rs: SVC(C=p["C"], kernel=p["kernel"], gamma=p["gamma"], class_weight=_cw(p["class_weight"]), random_state=rs),
        [Param("C", "Regularisation (C)", "float", 1.0, "Larger values fit the training data more tightly.", 1e-4, 1e4, log=True),
         Param("kernel", "Kernel", "choice", "rbf", "Shape of the decision boundary.", choices=["rbf", "linear", "poly", "sigmoid"]),
         Param("gamma", "Gamma", "choice", "scale", "Kernel coefficient.", choices=["scale", "auto"]),
         _class_weight()],
        aliases=["svm", "support vector machine", "svc"],
    ),
    Algorithm(
        "knn_classifier", "K-Nearest Neighbors", "classification", "Instance-based",
        "Predicts the majority class among the k closest training rows.",
        ["No training phase", "Naturally multi-class", "Intuitive"],
        ["Slow predictions on large data", "Degrades with many features", "Needs scaled features"],
        True,
        lambda p, rs: KNeighborsClassifier(n_neighbors=p["n_neighbors"], weights=p["weights"]),
        [Param("n_neighbors", "Neighbours (k)", "int", 5, "Number of neighbours that vote.", 1, 200),
         Param("weights", "Weighting", "choice", "uniform", "'distance' gives closer neighbours more weight.", choices=["uniform", "distance"])],
        aliases=["knn", "k-nearest neighbors", "nearest neighbors"],
    ),
    Algorithm(
        "gaussian_nb", "Gaussian Naive Bayes", "classification", "Probabilistic",
        "Applies Bayes' rule assuming features are independent and normally distributed per class.",
        ["Extremely fast", "Works with very little data"],
        ["Independence assumption rarely holds", "Probabilities are often poorly calibrated", "One-hot features violate the Gaussian assumption"],
        False,
        lambda p, rs: GaussianNB(var_smoothing=p["var_smoothing"]),
        [Param("var_smoothing", "Variance smoothing", "float", 1e-9, "Stabilises variance estimates.", 1e-12, 1.0, log=True)],
        aliases=["naive bayes", "gaussian naive bayes", "gaussiannb"],
    ),
    # --------------------------------------------------------- regression
    Algorithm(
        "linear_regression", "Linear Regression", "regression", "Linear",
        "Ordinary least squares fit of a weighted sum of features.",
        ["Fast, transparent baseline", "Coefficients are interpretable"],
        ["Only linear relationships", "Unstable with correlated features", "Sensitive to outliers"],
        True,
        lambda p, rs: LinearRegression(),
        [],
        aliases=["linear regression", "linear", "ols"],
    ),
    Algorithm(
        "ridge", "Ridge Regression", "regression", "Linear",
        "Linear regression with an L2 penalty that shrinks coefficients.",
        ["Stable with correlated features", "Fast"],
        ["Only linear relationships", "Needs scaled features for a fair penalty"],
        True,
        lambda p, rs: Ridge(alpha=p["alpha"], random_state=rs),
        [Param("alpha", "Alpha", "float", 1.0, "Penalty strength.", 1e-6, 1e5, log=True)],
        aliases=["ridge", "ridge regression"],
    ),
    Algorithm(
        "lasso", "Lasso Regression", "regression", "Linear",
        "Linear regression with an L1 penalty that can zero out features.",
        ["Built-in feature selection", "Sparse, interpretable models"],
        ["Picks one feature arbitrarily among correlated ones", "Needs scaled features"],
        True,
        lambda p, rs: Lasso(alpha=p["alpha"], max_iter=p["max_iter"], random_state=rs),
        [Param("alpha", "Alpha", "float", 0.1, "Penalty strength. Too large zeroes every coefficient.", 1e-6, 1e5, log=True),
         Param("max_iter", "Max iterations", "int", 5000, "Coordinate-descent budget.", 100, 100000)],
        aliases=["lasso", "lasso regression"],
    ),
    Algorithm(
        "decision_tree_regressor", "Decision Tree Regressor", "regression", "Tree",
        "A single tree predicting the mean target of each leaf.",
        ["Non-linear", "No scaling needed", "Interpretable"],
        ["Piecewise-constant predictions", "Overfits without depth limits"],
        False,
        lambda p, rs: DecisionTreeRegressor(max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"], random_state=rs),
        [_max_depth(8), _min_samples_leaf()],
        aliases=["decision tree regressor", "decision tree"],
    ),
    Algorithm(
        "random_forest_regressor", "Random Forest Regressor", "regression", "Ensemble",
        "Averages many decorrelated regression trees.",
        ["Strong default on tabular data", "Robust to outliers and scale"],
        ["Cannot extrapolate beyond the training target range", "Larger models"],
        False,
        lambda p, rs: RandomForestRegressor(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                                            min_samples_leaf=p["min_samples_leaf"], n_jobs=-1, random_state=rs),
        [_n_estimators(), _max_depth(), _min_samples_leaf()],
        aliases=["random forest regressor", "random forest", "rf"],
    ),
    Algorithm(
        "gradient_boosting_regressor", "Gradient Boosting Regressor", "regression", "Ensemble",
        "Sequentially fitted trees that correct residual errors.",
        ["Often the most accurate on small/medium data", "Captures interactions"],
        ["Slower to train", "Sensitive to learning rate"],
        False,
        lambda p, rs: GradientBoostingRegressor(n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
                                                max_depth=p["max_depth"], random_state=rs),
        [_n_estimators(150), Param("learning_rate", "Learning rate", "float", 0.1, "Shrinks each tree's contribution.", 0.001, 1.0, log=True),
         Param("max_depth", "Max depth", "int", 3, "Depth of each tree.", 1, 12)],
        aliases=["gradient boosting regressor", "gradient boosting", "gbm"],
    ),
    Algorithm(
        "hist_gradient_boosting_regressor", "Histogram Gradient Boosting Regressor", "regression", "Ensemble",
        "Fast binned gradient boosting for larger datasets.",
        ["Scales to large data", "Strong accuracy"],
        ["Less interpretable", "Overkill for tiny datasets"],
        False,
        lambda p, rs: HistGradientBoostingRegressor(max_iter=p["max_iter"], learning_rate=p["learning_rate"],
                                                    max_leaf_nodes=p["max_leaf_nodes"], random_state=rs),
        [Param("max_iter", "Boosting iterations", "int", 200, "Number of boosting rounds.", 10, 5000),
         Param("learning_rate", "Learning rate", "float", 0.1, "Shrinks each tree's contribution.", 0.001, 1.0, log=True),
         Param("max_leaf_nodes", "Max leaf nodes", "int", 31, "Leaves per tree.", 2, 512)],
        aliases=["hist gradient boosting regressor", "hist gradient boosting"],
    ),
    Algorithm(
        "svr", "Support Vector Regression", "regression", "Kernel",
        "Fits a function within an ε-tube around the targets, optionally with kernels.",
        ["Robust to some outliers", "Flexible with kernels"],
        ["Slow on large data", "Needs scaled features (and ideally a scaled target)"],
        True,
        lambda p, rs: SVR(C=p["C"], kernel=p["kernel"], epsilon=p["epsilon"]),
        [Param("C", "Regularisation (C)", "float", 1.0, "Larger values fit the data more tightly.", 1e-4, 1e5, log=True),
         Param("kernel", "Kernel", "choice", "rbf", "Shape of the function.", choices=["rbf", "linear", "poly"]),
         Param("epsilon", "Epsilon", "float", 0.1, "Width of the no-penalty tube, in target units.", 0.0, 1e6)],
        aliases=["svr", "support vector regression", "svm"],
    ),
    Algorithm(
        "knn_regressor", "K-Nearest Neighbors Regressor", "regression", "Instance-based",
        "Averages the targets of the k closest training rows.",
        ["Simple, non-parametric"],
        ["Slow predictions on large data", "Degrades with many features", "Needs scaled features"],
        True,
        lambda p, rs: KNeighborsRegressor(n_neighbors=p["n_neighbors"], weights=p["weights"]),
        [Param("n_neighbors", "Neighbours (k)", "int", 5, "Number of neighbours averaged.", 1, 200),
         Param("weights", "Weighting", "choice", "uniform", "'distance' gives closer neighbours more weight.", choices=["uniform", "distance"])],
        aliases=["knn", "knn regressor", "k-nearest neighbors"],
    ),
]

_xgb_reason = None if XGBOOST_AVAILABLE else "The 'xgboost' package is not installed on the server."
ALGORITHMS += [
    Algorithm(
        "xgboost_classifier", "XGBoost", "classification", "Ensemble",
        "Regularised gradient boosting from the XGBoost library.",
        ["State-of-the-art accuracy on many tabular problems", "Regularisation built in"],
        ["Many hyperparameters", "Extra dependency"],
        False,
        lambda p, rs: XGBClassifier(n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
                                    max_depth=p["max_depth"], random_state=rs, n_jobs=-1, verbosity=0),
        [_n_estimators(300), Param("learning_rate", "Learning rate", "float", 0.1, "Step size shrinkage.", 0.001, 1.0, log=True),
         Param("max_depth", "Max depth", "int", 6, "Depth of each tree.", 1, 16)],
        available=XGBOOST_AVAILABLE, unavailable_reason=_xgb_reason, aliases=["xgboost", "xgb"],
    ),
    Algorithm(
        "xgboost_regressor", "XGBoost Regressor", "regression", "Ensemble",
        "Regularised gradient boosting from the XGBoost library.",
        ["State-of-the-art accuracy on many tabular problems"],
        ["Many hyperparameters", "Extra dependency"],
        False,
        lambda p, rs: XGBRegressor(n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
                                   max_depth=p["max_depth"], random_state=rs, n_jobs=-1, verbosity=0),
        [_n_estimators(300), Param("learning_rate", "Learning rate", "float", 0.1, "Step size shrinkage.", 0.001, 1.0, log=True),
         Param("max_depth", "Max depth", "int", 6, "Depth of each tree.", 1, 16)],
        available=XGBOOST_AVAILABLE, unavailable_reason=_xgb_reason, aliases=["xgboost regressor", "xgboost", "xgb"],
    ),
]

BY_ID: Dict[str, Algorithm] = {a.id: a for a in ALGORITHMS}


def get(algorithm_id: str) -> Algorithm:
    try:
        return BY_ID[algorithm_id]
    except KeyError:
        raise AlgorithmError(f"Unknown algorithm '{algorithm_id}'. Supported: {', '.join(BY_ID)}.") from None


def resolve_name(name: str, task: str) -> Algorithm:
    """Map a legacy display name (old API) to a registered algorithm for ``task``.

    Raises instead of guessing when nothing matches.
    """
    key = (name or "").strip().lower()
    if key in BY_ID and BY_ID[key].task == task:
        return BY_ID[key]
    for alg in ALGORITHMS:
        if alg.task == task and (key == alg.name.lower() or key in alg.aliases):
            return alg
    supported = ", ".join(a.name for a in ALGORITHMS if a.task == task)
    raise AlgorithmError(f"'{name}' is not a supported {task} algorithm. Supported: {supported}.")


def for_task(task: Optional[str] = None) -> List[Algorithm]:
    return [a for a in ALGORITHMS if task is None or a.task == task]
