#!/usr/bin/env python
# coding: utf-8
"""
Out-of-fold evaluation + probability ensemble for multiclass text classification.
No Transformers: TF-IDF (word and/or character) + linear / naive Bayes models,
plus optional gradient boosting on concatenated TruncatedSVD features (word + char).

Ensemble averages calibrated probability vectors from diverse base learners to
often improve macro-F1 vs any single model.

`blended_predict_proba` centralizes the fold-wise / full-data weighted blend; use
`hierarchical_ensemble_pipeline.py` for hierarchical OOF vs this baseline on the same folds.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

SEED = 42
DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_SUBMISSIONS = Path(__file__).resolve().parent / "submissions"
OUT_META = Path(__file__).resolve().parent / "models"

# Tuned on OOF (4-fold): up-weight strong SGD-word; down-weight noisy NB a bit;
# GBDT on SVD features adds a non-linear view—slightly lower weight to avoid drowning linears.
DEFAULT_BLEND_WEIGHTS: dict[str, float] = {
    "sgd_word": 3.0,
    "sgd_word_char": 1.0,
    "complement_nb": 0.5,
    "linearsvc_calibrated": 1.0,
    "gbdt_tfidf_svd": 0.85,
}


def clean_text(text: str) -> str:
    """Clean and normalize a text string for use in text classification pipelines.

    This function performs several preprocessing steps to standardize raw text:
      - Converts input to lowercase.
      - Removes URLs and HTML tags.
      - Strips punctuation and non-alphanumeric characters (except whitespace).
      - Collapses multiple whitespace characters into a single space.
      - Trims leading/trailing spaces.

    Parameters
    ----------
    text : str
        The input text to clean. If the input is not a string (e.g., NaN, None),
        it is converted to an empty string.

    Returns
    -------
    str
        Cleaned and normalized text suitable for TF-IDF or other token-based models.
        Returns an empty string if input is invalid or becomes empty after cleaning.

    Notes
    -----
    - Designed for preprocessing user-generated text (e.g., social media, reviews).
    - Case folding improves vocabulary consistency for models without subword encoding.
    - URL and HTML tag removal reduces noise and prevents overfitting to web-specific patterns.
    - Punctuation removal simplifies feature space; may be suboptimal for models
      relying on emoticons or special symbols.
    - Extra whitespace is normalized to prevent spurious token separation.

    Example
    -------
    >>> clean_text("Check this out: https://example.com! Great product.")
    'check this out great product'

    >>> clean_text(None)
    ''

    This function is typically used within pandas `.map()` or `.apply()` operations
    during dataset preparation and must be applied consistently to train, validation,
    and test sets.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+|<.*?>", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()



def tfidf_word() -> TfidfVectorizer:
    """Create a TF-IDF vectorizer configured for word-level n-gram extraction.

    Returns a `TfidfVectorizer` instance tuned for text classification tasks,
    focusing on unigrams and bigrams with preprocessing optimizations to reduce noise
    and improve generalization.

    Returns
    -------
    TfidfVectorizer
        Configured vectorizer with the following parameters:
        - `max_features=60000`: Limits vocabulary size to top 60k features by TF-IDF score.
        - `ngram_range=(1, 2)`: Includes both unigrams (single words) and bigrams (word pairs).
        - `min_df=2`: Ignores terms that appear in fewer than 2 documents.
        - `max_df=0.95`: Filters out terms appearing in more than 95% of documents (common stopwords).
        - `sublinear_tf=True`: Applies log scaling to term frequencies (tf → 1 + log(tf)).
        - `strip_accents="unicode"`: Normalizes accented characters (e.g., café → cafe).
        - `dtype=np.float32`: Reduces memory usage while maintaining numerical stability.

    Notes
    -----
    - Suitable for models expecting dense numerical input from raw text.
    - Unicode stripping improves token matching across different character encodings.
    - Sublinear TF helps dampen the effect of very frequent terms within documents.
    - The combination of `min_df` and `max_df` enhances robustness by removing rare and overly common terms.
    - Designed to work seamlessly with `Pipeline` and `GridSearchCV`.

    Example
    -------
    >>> vectorizer = tfidf_word()
    >>> X = vectorizer.fit_transform(["hello world", "machine learning"])
    >>> X.shape
    (2, N)  # where N <= 60000

    This configuration balances expressiveness and efficiency, making it ideal for
    baseline and ensemble text classifiers such as SGD, SVM, or Naive Bayes.
    """
    return TfidfVectorizer(
        max_features=60000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents="unicode",
        dtype=np.float32,
    )



def _svd_n_components(requested: int, n_samples: int, n_features: int) -> int:
    """Determine safe number of SVD components based on data dimensions.

    Computes the maximum valid number of singular value decomposition (SVD) components
    given the input data shape, ensuring it does not exceed the theoretical limit
    imposed by linear algebra constraints.

    Parameters
    ----------
    requested : int
        The desired number of SVD components.
    n_samples : int
        Number of samples (rows) in the input data matrix.
    n_features : int
        Number of features (columns) in the input data matrix.

    Returns
    -------
    int
        The actual number of SVD components to use, bounded by:
        - Minimum of 1.
        - Maximum of `min(n_features - 1, n_samples - 1)` due to rank limitation.
        - Clamped to `requested` if within valid range.

    Notes
    -----
    - The rank of a matrix cannot exceed `min(n_samples, n_features)`, so the number
      of meaningful SVD components is at most `min(n_samples, n_features) - 1` when
      centering is applied (e.g., in TruncatedSVD or PCA).
    - Returns 1 if either dimension is invalid (<= 1) to prevent errors.
    - Used internally to avoid `ValueError` from decomposition algorithms when
      requesting too many components.

    Example
    -------
    >>> _svd_n_components(100, n_samples=500, n_features=200)
    199
    >>> _svd_n_components(50, n_samples=1000, n_features=10000)
    50

    This utility is typically used in preprocessing pipelines involving dimensionality
    reduction (e.g., LSA/TruncatedSVD) where feature counts vary dynamically.
    """
    if n_features <= 1 or n_samples <= 1:
        return 1
    cap = min(n_features - 1, n_samples - 1)
    return max(1, min(int(requested), int(cap)))



class TfidfDualSvdConcat(BaseEstimator, TransformerMixin):
    """
    Word TF-IDF -> SVD and char-wb TF-IDF -> SVD; horizontally stack dense outputs.
    Fits separate vocabularies and SVDs on the training fold only (pipeline-safe).
    """

    def __init__(
        self,
        *,
        word_max_features: int = 50_000,
        char_max_features: int = 60_000,
        n_components_word: int = 180,
        n_components_char: int = 180,
        random_state: int = SEED,
    ):
        """Initialize the transformer with configurable TF-IDF and SVD parameters.

        This transformer extracts two types of features from text:
          - Word-level n-grams (unigrams and bigrams) via TF-IDF.
          - Character-level n-grams ('char_wb' mode, 3–5 grams) via TF-IDF.
        Each stream is independently dimensionality-reduced using TruncatedSVD,
        and the resulting dense vectors are concatenated.

        Parameters
        ----------
        word_max_features : int, default=50_000
            Maximum number of features for the word-level TF-IDF vectorizer.
        char_max_features : int, default=60_000
            Maximum number of features for the character-level TF-IDF vectorizer.
        n_components_word : int, default=180
            Target number of components for word SVD. Will be capped at `min(n_samples, n_features) - 1`.
        n_components_char : int, default=180
            Target number of components for character SVD. Will be capped similarly.
        random_state : int, default=SEED
            Random seed for reproducible SVD results.

        Notes
        -----
        - Designed to be used within scikit-learn pipelines and CV splits.
        - Fitting on a cross-validation fold ensures no data leakage.
        - Uses 'char_wb' analyzer to generate character n-grams only within word boundaries,
          improving interpretability and reducing noise.
        - Sublinear TF scaling and document frequency cutoffs improve robustness.
        - Output is a dense `float32` array suitable for input to linear or tree-based models.

        Example
        -------
        >>> transformer = TfidfDualSvdConcat(n_components_word=100, n_components_char=120)
        >>> X = ["this is a test", "another example"]
        >>> Xt = transformer.fit_transform(X)
        >>> Xt.shape
        (2, 220)  # 100 + 120
        """
        self.word_max_features = word_max_features
        self.char_max_features = char_max_features
        self.n_components_word = n_components_word
        self.n_components_char = n_components_char
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the word and char TF-IDF vectorizers and their respective SVD projectors.

        Learns vocabulary and term weights from `X`, then computes optimal low-rank
        projections for both feature spaces.

        Parameters
        ----------
        X : array-like of str, shape (n_samples,)
            Training text data.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : TfidfDualSvdConcat
            Fitted transformer instance.

        Notes
        -----
        - Both vectorizers are fitted on the same input `X`.
        - The actual number of SVD components is adjusted using `_svd_n_components`
          to ensure it does not exceed rank limits.
        - All fitted components (vectorizers and SVDs) are stored as private attributes.
        """
        self.word_vec_ = TfidfVectorizer(
            max_features=self.word_max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            strip_accents="unicode",
            dtype=np.float32,
        )
        self.char_vec_ = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=self.char_max_features,
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            dtype=np.float32,
        )
        X_w = self.word_vec_.fit_transform(X)
        X_c = self.char_vec_.fit_transform(X)
        n_w = _svd_n_components(self.n_components_word, X_w.shape[0], X_w.shape[1])
        n_c = _svd_n_components(self.n_components_char, X_c.shape[0], X_c.shape[1])
        self.svd_w_ = TruncatedSVD(n_components=n_w, random_state=self.random_state)
        self.svd_c_ = TruncatedSVD(n_components=n_c, random_state=self.random_state)
        self.svd_w_.fit(X_w)
        self.svd_c_.fit(X_c)
        return self

    def transform(self, X):
        """Transform `X` into a dense concatenated feature vector.

        Applies both fitted TF-IDF vectorizers and SVD projectors, then concatenates
        the resulting low-dimensional representations.

        Parameters
        ----------
        X : array-like of str, shape (n_samples,)
            Input text data to transform.

        Returns
        -------
        np.ndarray of shape (n_samples, n_components_word + n_components_char)
            Dense float32 array containing stacked SVD projections.
            Actual number of columns may be less if component count was capped during fit.

        Notes
        -----
        - Uses `transform` (not `fit_transform`) on vectorizers and SVDs to prevent refitting.
        - Output is memory-efficient with `copy=False` where possible.
        - Ideal for use as a feature extractor in ensemble methods or stacking pipelines.
        """
        X_w = self.word_vec_.transform(X)
        X_c = self.char_vec_.transform(X)
        z_w = self.svd_w_.transform(X_w)
        z_c = self.svd_c_.transform(X_c)
        return np.hstack([z_w, z_c]).astype(np.float32, copy=False)



def _make_gbdt_svd_classifier():
    """LightGBM if installed; else sklearn HistGradientBoosting (multiclass).

    Constructs a gradient-boosted decision tree classifier suitable for multiclass
    classification tasks. Prefers LightGBM for speed and performance if available;
    falls back to scikit-learn's HistGradientBoostingClassifier otherwise.

    Returns
    -------
    estimator : LGBMClassifier or HistGradientBoostingClassifier
        A configured GBDT model with reasonable defaults for moderate-sized datasets.
        - If `lightgbm` is installed: returns `LGBMClassifier` with parameters tuned
          for stability and generalization.
        - Else: returns `HistGradientBoostingClassifier` with comparable settings.

    Notes
    -----
    - Uses `class_weight="balanced"` in both cases to handle potential label imbalance.
    - All models are seeded with `SEED` for reproducibility.
    - LightGBM settings:
        - `n_estimators=700`, `learning_rate=0.05`: Moderate boosting rounds with shrinkage.
        - `num_leaves=48`: Limits tree complexity.
        - Subsampling (`subsample`, `colsample_bytree`) improves robustness.
        - Regularization via `reg_lambda=1.0`.
        - Silent output with `verbosity=-1`.
    - Fallback HistGradientBoosting settings:
        - `max_iter=450`: Maximum number of boosting iterations.
        - `max_depth=11`, `min_samples_leaf=12`: Controls overfitting.
        - `l2_regularization=1.0`: Equivalent to LightGBM's lambda.
        - Early stopping enabled with validation split and patience of 25 rounds.
    - Designed for use with dense numerical features (e.g., SVD-reduced TF-IDF).

    Example
    -------
    >>> clf = _make_gbdt_svd_classifier()
    >>> clf.fit(X_train, y_train)
    >>> preds = clf.predict(X_test)

    This function enables portable code that leverages high-performance GBDTs when
    available but remains functional in minimal environments (e.g., CI/CD, base Python).
    """
    try:
        import lightgbm as lgb  # type: ignore[import-not-found]

        return lgb.LGBMClassifier(
            objective="multiclass",
            class_weight="balanced",
            n_estimators=700,
            learning_rate=0.05,
            num_leaves=48,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_samples=15,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=SEED,
            n_jobs=-1,
            verbosity=-1,
        )
    except ImportError:
        return HistGradientBoostingClassifier(
            learning_rate=0.07,
            max_iter=450,
            max_depth=11,
            min_samples_leaf=12,
            l2_regularization=1.0,
            max_bins=255,
            class_weight="balanced",
            random_state=SEED,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=25,
        )



def make_gbdt_tfidf_svd_pipeline() -> Pipeline:
    """Create a complete classification pipeline combining TF-IDF, SVD, and GBDT.

    Constructs a scikit-learn `Pipeline` that performs:
      1. Dual TF-IDF vectorization (word + character n-grams).
      2. Dimensionality reduction via TruncatedSVD for both streams.
      3. Concatenation of reduced features.
      4. Classification using a gradient-boosted decision tree (LightGBM or fallback).

    Returns
    -------
    Pipeline
        A two-stage pipeline:
        - 'prep': Instance of `TfidfDualSvdConcat` configured with:
            - Word TF-IDF: up to 50k features, unigrams/bigrams.
            - Char TF-IDF: up to 60k features, 3–5 char n-grams (`char_wb` mode).
            - SVD: 180 components each (capped by rank), random state fixed.
        - 'clf': Output of `_make_gbdt_svd_classifier()` — either `LGBMClassifier`
          or `HistGradientBoostingClassifier`, depending on availability.

    Notes
    -----
    - Designed for text classification with robustness to spelling variations
      (via char n-grams) and efficient dense feature representation.
    - Fully pipeline-compatible: supports `fit`, `predict`, `predict_proba`, and cross-validation.
    - All stages are fitted only on training data, preventing leakage.
    - Uses consistent `SEED` for reproducibility across runs.
    - Ideal for medium-to-large text datasets where interpretability is less critical
      than predictive performance.

    Example
    -------
    >>> pipe = make_gbdt_tfidf_svd_pipeline()
    >>> pipe.fit(X_train, y_train)
    >>> predictions = pipe.predict(X_test)

    This pipeline represents a strong baseline or ensemble component in text modeling
    competitions and production systems, balancing accuracy and generalization.
    """
    return Pipeline(
        [
            (
                "prep",
                TfidfDualSvdConcat(
                    word_max_features=50_000,
                    char_max_features=60_000,
                    n_components_word=180,
                    n_components_char=180,
                    random_state=SEED,
                ),
            ),
            ("clf", _make_gbdt_svd_classifier()),
        ]
    )



def gbdt_backend_name() -> str:
    """Determine the name of the GBDT backend that will be used by the pipeline.

    Checks whether LightGBM is available in the current environment.
    Returns the corresponding backend identifier used for logging and configuration.

    Returns
    -------
    str
        - "lightgbm" if the `lightgbm` package is installed.
        - "hist_gradient_boosting" if LightGBM is not available and scikit-learn's
          `HistGradientBoostingClassifier` will be used as fallback.

    Notes
    -----
    - Used to document model provenance in metadata (e.g., tracking which GBDT engine
      was active during training).
    - Relies on the presence of the `lightgbm` module; does not check version or capabilities.
    - The return value aligns with the classifier returned by `_make_gbdt_svd_classifier`.
    - Import is suppressed (`noqa: F401`) because it's only used for availability checking.

    Example
    -------
    >>> gbdt_backend_name()
    'lightgbm'

    This function helps ensure reproducibility and transparency in ensemble pipelines
    where the underlying model may vary based on environment dependencies.
    """
    try:
        import lightgbm  # noqa: F401

        return "lightgbm"
    except ImportError:
        return "hist_gradient_boosting"



def tfidf_word_char_union() -> FeatureUnion:
    """Create a feature union of word and character-level TF-IDF vectorizers.

    Constructs a `FeatureUnion` that combines two independently configured `TfidfVectorizer`
    instances:
      - One for word n-grams (unigrams and bigrams).
      - One for character n-grams using word boundaries ('char_wb').

    Returns
    -------
    FeatureUnion
        A composite transformer that applies both vectorizers in parallel and concatenates
        their outputs into a single sparse matrix. The resulting features are the union of:
        - Word-based TF-IDF features (with preprocessing: Unicode normalization, sublinear TF).
        - Character n-gram TF-IDF features (3–5 characters long, within word boundaries).

    Notes
    -----
    - Both vectorizers are limited to 45,000 features to control memory usage.
    - `min_df=2` and `max_df=0.95` filter out rare and overly frequent terms.
    - `sublinear_tf=True` applies log scaling to term frequencies: tf → 1 + log(tf).
    - `strip_accents="unicode"` normalizes accented characters for better matching.
    - Using `analyzer="char_wb"` ensures character n-grams are only formed within word boundaries,
      reducing noise from跨word sequences.
    - Output is a sparse matrix suitable for use with linear models or dimensionality reduction.

    Example
    -------
    >>> X = ["hello world", "machine learning"]
    >>> transformer = tfidf_word_char_union()
    >>> Xt = transformer.fit_transform(X)
    >>> Xt.shape
    (2, N)  # where N <= 90000 (45k word + 45k char)

    This transformer is ideal for text classification tasks where robustness to spelling
    variations, morphology, and domain-specific jargon is important.
    """
    word = TfidfVectorizer(
        max_features=45000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents="unicode",
        dtype=np.float32,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=45000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        dtype=np.float32,
    )
    return FeatureUnion([("word", word), ("char", char)])



def align_proba(
    proba: np.ndarray, model_classes: np.ndarray, ref_classes: np.ndarray
) -> np.ndarray:
    """Map predict_proba columns to ref_classes order (handles missing classes in a fold).

    Reorders the probability array output by `predict_proba` so that its columns match
    the class ordering of a reference label encoder (`ref_classes`). This is essential
    when combining predictions from models trained on different CV folds where some
    classes may be missing, leading to mismatched column indices.

    Parameters
    ----------
    proba : np.ndarray of shape (n_samples, n_model_classes)
        Probability predictions from a classifier's `predict_proba` method.
    model_classes : np.ndarray of shape (n_model_classes,)
        Class labels as known by the model (i.e., `model.classes_`).
    ref_classes : np.ndarray of shape (n_ref_classes,)
        Reference class labels (e.g., from a global LabelEncoder), defining the desired output order.

    Returns
    -------
    np.ndarray of shape (n_samples, n_ref_classes)
        Reordered probability array where column `k` corresponds to `ref_classes[k]`.
        If a class in `ref_classes` was not present during model training (and thus not in `model_classes`),
        the corresponding column is filled with zeros.

    Notes
    -----
    - Ensures compatibility across models with potentially different sets of seen classes.
    - Critical for ensembling or stacking pipelines using out-of-fold predictions.
    - Uses `int(c)` to ensure consistent comparison even if class labels are numeric as strings.
    - Output dtype is `float64` for numerical stability in downstream operations.

    Example
    -------
    >>> proba = np.array([[0.8, 0.2], [0.3, 0.7]])  # model predicts classes [0, 1]
    >>> model_classes = np.array([0, 1])
    >>> ref_classes = np.array([0, 1, 2])
    >>> aligned = align_proba(proba, model_classes, ref_classes)
    >>> aligned.shape
    (2, 3)
    >>> aligned[:, 2]  # probabilities for class 2 (missing in model) are zero
    array([0., 0.])

    This function enables safe aggregation of probabilistic predictions in multi-class
    classification pipelines, especially under imbalanced or stratified CV schemes.
    """
    n_samples = proba.shape[0]
    out = np.zeros((n_samples, len(ref_classes)), dtype=np.float64)
    model_to_j = {int(c): j for j, c in enumerate(model_classes)}
    for k, c in enumerate(ref_classes):
        j = model_to_j.get(int(c))
        if j is not None:
            out[:, k] = proba[:, j]
    return out



def make_base_learners(
    *,
    use_calibrated_svc: bool = True,
    calibrated_svc_cv: int = 2,
    use_gbdt_svd: bool = True,
) -> list[tuple[str, Pipeline | CalibratedClassifierCV]]:
    """Create a list of base learners with predict_proba support for ensemble modeling.

    Constructs multiple heterogeneous classifiers wrapped in pipelines, each capable
    of producing well-calibrated probability estimates via `predict_proba`. These
    models are designed to be used in stacking, blending, or voting ensembles.

    Parameters
    ----------
    use_calibrated_svc : bool, default=True
        Whether to include a calibrated LinearSVC model. Since `LinearSVC` lacks
        `predict_proba`, it is wrapped in `CalibratedClassifierCV` for probability output.
    calibrated_svc_cv : int, default=2
        Number of cross-validation folds to use when calibrating the SVC model.
        Lower values reduce training time at potential cost to calibration quality.
    use_gbdt_svd : bool, default=True
        Whether to include the GBDT + TF-IDF-SVD pipeline, which combines dual
        vectorization (word and char), SVD compression, and gradient boosting.

    Returns
    -------
    list[tuple[str, Pipeline | CalibratedClassifierCV]]
        List of (name, estimator) pairs suitable for use in ensemble methods.
        All estimators support `predict_proba`. The returned learners are:
        - 'sgd_word': SGDClassifier on word-level TF-IDF (unigrams/bigrams).
        - 'sgd_word_char': SGDClassifier on combined word + char-wb TF-IDF.
        - 'complement_nb': Complement Naive Bayes on word TF-IDF (good for imbalanced text).
        - 'linearsvc_calibrated' (optional): Calibrated LinearSVC on word TF-IDF.
        - 'gbdt_tfidf_svd' (optional): GBDT classifier on SVD-compressed dual TF-IDF features.

    Notes
    -----
    - LogisticRegression is intentionally omitted due to its slow convergence on large TF-IDF matrices.
    - All models use `class_weight="balanced"` where applicable to handle label imbalance.
    - Random states are fixed using `SEED` for reproducibility.
    - The `tfidf_word_char_union()` and `make_gbdt_tfidf_svd_pipeline()` functions are reused
      to ensure consistent preprocessing.
    - Calibrated models use `method="sigmoid"` (Platt scaling) and are parallelized (`n_jobs=-1`).

    Example
    -------
    >>> learners = make_base_learners(use_gbdt_svd=False)
    >>> len(learners)
    4
    >>> names = [name for name, _ in learners]
    >>> "sgd_word" in names
    True

    This function provides a modular way to define an ensemble of diverse, high-performing
    text classifiers, balancing speed, accuracy, and generalization across different data regimes.
    """
    learners: list[tuple[str, Pipeline | CalibratedClassifierCV]] = []

    learners.append(
        (
            "sgd_word",
            Pipeline(
                [
                    ("vec", tfidf_word()),
                    (
                        "clf",
                        SGDClassifier(
                            loss="log_loss",
                            penalty="l2",
                            alpha=1e-4,
                            max_iter=3000,
                            class_weight="balanced",
                            random_state=SEED,
                            n_jobs=-1,
                            tol=1e-3,
                        ),
                    ),
                ]
            ),
        )
    )

    learners.append(
        (
            "sgd_word_char",
            Pipeline(
                [
                    ("vec", tfidf_word_char_union()),
                    (
                        "clf",
                        SGDClassifier(
                            loss="log_loss",
                            penalty="l2",
                            alpha=1e-4,
                            max_iter=3000,
                            class_weight="balanced",
                            random_state=SEED,
                            n_jobs=-1,
                            tol=1e-3,
                        ),
                    ),
                ]
            ),
        )
    )

    learners.append(
        (
            "complement_nb",
            Pipeline(
                [
                    ("vec", TfidfVectorizer(**tfidf_word().get_params())),
                    ("clf", ComplementNB(alpha=0.02)),
                ]
            ),
        )
    )

    if use_calibrated_svc:
        base_svc = Pipeline(
            [
                ("vec", TfidfVectorizer(**tfidf_word().get_params())),
                (
                    "inner",
                    LinearSVC(
                        C=0.5,
                        class_weight="balanced",
                        random_state=SEED,
                        max_iter=5000,
                        dual="auto",
                    ),
                ),
            ]
        )
        learners.append(
            (
                "linearsvc_calibrated",
                CalibratedClassifierCV(
                    base_svc,
                    method="sigmoid",
                    cv=calibrated_svc_cv,
                    n_jobs=-1,
                ),
            )
        )

    if use_gbdt_svd:
        learners.append(("gbdt_tfidf_svd", make_gbdt_tfidf_svd_pipeline()))

    return learners



def blended_predict_proba(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: np.ndarray,
    ref_classes: np.ndarray,
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    """
    Fit each base learner on (X_train, y_train) and return the weighted blend of
    predict_proba on X_query, aligned to ref_classes columns (same recipe as OOF folds).

    Trains multiple base classifiers independently on the provided training data,
    generates calibrated probability estimates for the query set, aligns them to a
    common class ordering, and returns a weighted average of the predictions.

    Parameters
    ----------
    X_train : np.ndarray of shape (n_train_samples,)
        Training text data (typically raw strings).
    y_train : np.ndarray of shape (n_train_samples,)
        Training labels.
    X_query : np.ndarray of shape (n_query_samples,)
        Query text data to predict probabilities for.
    ref_classes : np.ndarray of shape (n_classes,)
        Reference class labels defining the output column order. Ensures consistency
        across different calls, especially when some models may not see all classes.
    weights : dict[str, float], optional
        Mapping from learner name (e.g., 'sgd_word') to its weight in the final blend.
        If None, all learners are equally weighted.

    Returns
    -------
    np.ndarray of shape (n_query_samples, n_classes)
        Weighted average of predicted probabilities from all base learners,
        with columns aligned to `ref_classes`. The output is normalized such that
        probabilities sum to 1.0 per sample (up to floating point precision).

    Notes
    -----
    - Uses `make_base_learners()` to obtain a list of (name, estimator) pipelines.
      All estimators must support `predict_proba`.
    - Each model is cloned before fitting to avoid modifying shared templates.
    - Class alignment via `align_proba` ensures robust handling of missing classes
      in certain folds (common in stratified CV with rare labels).
    - Weights are normalized by their sum to produce a convex combination.
    - A small epsilon (`1e-9`) prevents division by zero if all weights are zero.
    - Ideal for generating out-of-fold (OOF) or test-set predictions in ensemble pipelines.

    Example
    -------
    >>> X_train = np.array(["good movie", "bad acting"])
    >>> y_train = np.array([1, 0])
    >>> X_query = np.array(["excellent film", "terrible plot"])
    >>> ref_classes = np.array([0, 1])
    >>> proba = blended_predict_proba(X_train, y_train, X_query, ref_classes)
    >>> proba.shape
    (2, 2)
    >>> np.allclose(proba.sum(axis=1), 1.0)
    True

    This function implements a core component of model stacking and soft voting,
    combining diverse classifiers to improve generalization and calibration.
    """
    learners_spec = make_base_learners()
    if weights is None:
        weights = {name: 1.0 for name, _ in learners_spec}
    w_sum = sum(float(weights.get(name, 1.0)) for name, _ in learners_spec)
    acc = np.zeros((len(X_query), len(ref_classes)), dtype=np.float64)
    for name, est in learners_spec:
        m = clone(est)
        m.fit(X_train, y_train)
        p = m.predict_proba(X_query)
        cls = getattr(m, "classes_", None)
        if cls is None:
            raise RuntimeError(f"No classes_ on {name}")
        p_aligned = align_proba(p, cls, ref_classes)
        acc += (float(weights.get(name, 1.0)) / max(w_sum, 1e-9)) * p_aligned
    return acc



def oof_proba_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    weights: dict[str, float] | None = None,
) -> tuple[np.ndarray, list[float], np.ndarray]:
    """
    Build out-of-fold stacked probability predictions on training data.
    Returns (oof_proba, fold_macro_f1_scores, ref_classes).

    Generates calibrated out-of-fold (OOF) probabilistic predictions using a weighted
    ensemble of base learners. Each fold trains on a subset of the data and predicts
    on the held-out portion, ensuring no data leakage. This OOF matrix can be used
    for model evaluation, stacking, or meta-learning.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples,)
        Input text data (typically raw strings).
    y : np.ndarray of shape (n_samples,)
        Ground truth labels.
    n_splits : int
        Number of folds in the stratified k-fold splitting strategy.
    weights : dict[str, float], optional
        Custom weights for base learners in the blend. Keys are learner names
        (e.g., 'sgd_word'), values are positive floats. If None, all learners
        are equally weighted.

    Returns
    -------
    oof_proba : np.ndarray of shape (n_samples, n_classes)
        Out-of-fold predicted probabilities aligned to `ref_classes`. Each row
        corresponds to the prediction made when that sample was in the validation fold.
    fold_macro_f1_scores : list[float]
        Macro-averaged F1 score for each fold's predictions, useful for assessing
        fold-wise model stability.
    ref_classes : np.ndarray of shape (n_classes,)
        Sorted unique class labels derived from `y`, defining the column order
        of the output probability matrix.

    Notes
    -----
    - Uses `StratifiedKFold` with fixed `SEED` to ensure reproducible splits.
    - For each fold, calls `blended_predict_proba` to combine predictions from
      multiple base learners trained on the training split.
    - Class alignment via `ref_classes` ensures consistent column ordering across folds.
    - Prediction quality is evaluated using macro F1 (handles class imbalance).
    - Zero-division in F1 is handled gracefully (returns 0.0 if no predictions are made).
    - The returned `oof` array is fully compatible with second-level meta-classifiers.

    Example
    -------
    >>> X = np.array(["great film", "bad acting", "awesome movie", "terrible"])
    >>> y = np.array([1, 0, 1, 0])
    >>> oof_pred, scores, classes = oof_proba_ensemble(X, y, n_splits=2)
    >>> oof_pred.shape
    (4, 2)
    >>> len(scores)
    2

    This function implements the first stage of a stacked generalization pipeline,
    producing robust, de-biased probability estimates suitable for both evaluation
    and use as features in higher-level models.
    """
    ref_classes = np.sort(np.unique(y))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.zeros((len(y), len(ref_classes)), dtype=np.float64)
    fold_scores: list[float] = []

    if weights is None:
        learners_spec = make_base_learners()
        weights = {name: 1.0 for name, _ in learners_spec}

    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_va = y[va_idx]

        blend_va = blended_predict_proba(X_tr, y[tr_idx], X_va, ref_classes, weights)
        oof[va_idx] = blend_va
        y_pred = ref_classes[np.argmax(blend_va, axis=1)]
        fold_scores.append(f1_score(y_va, y_pred, average="macro", zero_division=0))

    return oof, fold_scores, ref_classes



def main() -> None:
    """Execute the full training, validation, and submission pipeline for a text classification ensemble.

    Orchestrates the entire workflow:
      1. Load and preprocess train/test data.
      2. Determine cross-validation strategy based on dataset sizes.
      3. Generate out-of-fold (OOF) predictions using a weighted ensemble of base learners.
      4. Evaluate OOF performance via macro F1-score.
      5. Retrain all models on full training data and predict on test set.
      6. Produce a submission file and metadata log.

    The ensemble uses probability averaging (`blended_predict_proba`) from diverse
    base classifiers (SGD, ComplementNB, calibrated SVC, GBDT-SVD), with class alignment
    and fold consistency ensured throughout.

    Workflow Steps
    --------------
    - Loads `train.csv`, `test.csv`, and `sample_submission.csv`.
    - Applies `clean_text` to all text entries.
    - Sets number of CV folds so that validation size approximates test set size.
    - Reports which GBDT backend is active (LightGBM or HistGradientBoosting).
    - Computes OOF predictions using `oof_proba_ensemble` with predefined `DEFAULT_BLEND_WEIGHTS`.
    - Prints aggregated and per-fold macro F1 scores.
    - Refits the weighted ensemble on full training data to predict test labels.
    - Ensures submission ID order matches sample submission (fallback mapping if needed).
    - Saves:
        - Final submission CSV to `OUT_SUBMISSIONS/ensemble_oof_blended_submission.csv`.
        - Metadata JSON (config, scores, paths) to `OUT_META/multiclass_ensemble_cv.json`.

    Side Effects
    ------------
    - Writes files to disk (submission and metadata).
    - Prints progress, timing, and performance metrics to stdout.

    Notes
    -----
    - Designed to be run as a standalone script or entry point.
    - Uses global constants: `DATA_DIR`, `OUT_SUBMISSIONS`, `OUT_META`, `SEED`, `DEFAULT_BLEND_WEIGHTS`.
    - Assumes `clean_text` is defined elsewhere and handles text normalization.
    - Handles potential ID misalignment between `test_df` and `sample_sub`.
    - Total execution time is logged in metadata.

    Example Output
    --------------
    Train=10000 Test=5000 n_splits=2 (~5000 val size ≈ test)
    GBDT branch (TF-IDF dual SVD): lightgbm
    OOF blended macro-F1 (full train, stacked): 0.8742
    Per-fold macro-F1: [0.8713, 0.8771]
    Refitting all base learners on full training data (blended test predict)...
      blended_predict_proba in 12.4s
    Submission: submissions/ensemble_oof_blended_submission.csv
    Metadata: meta/multiclass_ensemble_cv.json
    Total time: 89.3s

    This function serves as the production inference pipeline, combining robust
    cross-validation evaluation with final model deployment for test prediction.
    """
    t0 = time.perf_counter()
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

    X = train_df["text"].map(clean_text).values
    y = train_df["target"].values
    X_test = test_df["text"].map(clean_text).values

    n_test = len(test_df)
    n_splits = max(2, int(round(len(train_df) / n_test)))

    print(
        f"Train={len(train_df)} Test={n_test} n_splits={n_splits} "
        f"(~{len(train_df)//n_splits} val size ≈ test)"
    )
    print(f"GBDT branch (TF-IDF dual SVD): {gbdt_backend_name()}")

    blend_weights = DEFAULT_BLEND_WEIGHTS
    oof, fold_f1, ref_classes = oof_proba_ensemble(
        X, y, n_splits=n_splits, weights=blend_weights
    )
    y_oof = ref_classes[np.argmax(oof, axis=1)]
    oof_macro = f1_score(y, y_oof, average="macro", zero_division=0)
    print(f"OOF blended macro-F1 (full train, stacked): {oof_macro:.4f}")
    print(f"Per-fold macro-F1: {[round(x, 4) for x in fold_f1]}")

    print("Refitting all base learners on full training data (blended test predict)...")
    learners_spec = make_base_learners()
    weights = {name: float(blend_weights.get(name, 1.0)) for name, _ in learners_spec}
    t1 = time.perf_counter()
    test_blend = blended_predict_proba(X, y, X_test, ref_classes, weights)
    print(f"  blended_predict_proba in {time.perf_counter() - t1:.1f}s")

    test_pred = ref_classes[np.argmax(test_blend, axis=1)].astype(int)

    submission = pd.DataFrame({"id": test_df["id"], "target": test_pred})
    if list(sample_sub["id"]) != list(submission["id"]):
        pred_map = dict(zip(test_df["id"].values, test_pred, strict=True))
        submission = sample_sub[["id"]].copy()
        submission["target"] = submission["id"].map(pred_map).astype(int)

    OUT_SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    OUT_META.mkdir(parents=True, exist_ok=True)
    sub_path = OUT_SUBMISSIONS / "ensemble_oof_blended_submission.csv"
    submission[["id", "target"]].to_csv(sub_path, index=False)

    meta = {
        "method": "weighted_average_predict_proba_oof_eval",
        "blend_weights": weights,
        "gbdt_backend": gbdt_backend_name(),
        "base_learners": [n for n, _ in learners_spec],
        "cv_n_splits": n_splits,
        "oof_macro_f1_full_train": float(oof_macro),
        "oof_fold_macro_f1": [float(x) for x in fold_f1],
        "submission_path": str(sub_path),
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
    }
    meta_path = OUT_META / "multiclass_ensemble_cv.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Submission: {sub_path}")
    print(f"Metadata: {meta_path}")
    print(f"Total time: {meta['elapsed_seconds']:.1f}s")



if __name__ == "__main__":
    main()
