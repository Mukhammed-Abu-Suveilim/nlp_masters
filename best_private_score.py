#!/usr/bin/env python
# coding: utf-8
"""
Final multiclass text-classification pipeline.

What this does (in order)
-------------------------
1. Load CSVs, clean text (NFKC + URL/USER/NAME placeholders; keeps punctuation for char n-grams).
2. Stratified K-fold OOF: fold count ≈ train/test size ratio (same idea as before).
3. For each fold, fit each base learner on train, predict probabilities on validation.
4. Combine base learners with fixed positive weights (tuned previously on OOF).
5. Optional: fit a meta logistic regression on stacked OOF probabilities; if it beats (4) on OOF
   macro-F1, use it for test; otherwise keep (4).
6. Greedy per-class log-bias search on the chosen OOF probabilities to improve macro-F1.
7. Refit every base learner on full training data, blend (or meta) on test, apply biases, write submission.

Base learners (same family as multiclass_ensemble_cv, best public performer):
  - SGD + word TF-IDF
  - SGD + word+char TF-IDF union
  - ComplementNB + word TF-IDF
  - Calibrated LinearSVC + word TF-IDF
  - Gradient boosting on dual TF-IDF + TruncatedSVD dense features (HGB or LightGBM)

Run:
  python final_pipeline.py
  python final_pipeline.py --no-meta     # skip meta stack, only blend + bias
  python final_pipeline.py --quick-bias  # fewer bias iterations (faster)
"""
from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

# ---------------------------------------------------------------------------
# Paths & reproducibility
# ---------------------------------------------------------------------------
SEED = 42
DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_SUBMISSIONS = Path(__file__).resolve().parent / "submissions"
OUT_META = Path(__file__).resolve().parent / "models"

# Weights: from OOF tuning on the previous ensemble (sgd_word up-weighted, NB down-weighted).
BLEND_WEIGHTS: dict[str, float] = {
    "sgd_word": 3.0,
    "sgd_word_char": 1.0,
    "complement_nb": 0.5,
    "linearsvc_calibrated": 1.0,
    "gbdt_tfidf_svd": 0.85,
}

# Slightly looser max_df (often helps rare tokens vs 0.95. Tried 0.85 as well).
WORD_MAX_DF = 0.985


def clean_text(text: str) -> str:
    """Apply comprehensive text cleaning suitable for NLP models.

    This function performs multiple cleaning steps to normalize and sanitize text
    while preserving meaningful symbols such as '#' and '!', which may carry
    semantic value for machine learning models. It replaces URLs, user mentions,
    and placeholders, unescapes HTML entities, and ensures uniform whitespace.

    Args:
        text (str): The input text to be cleaned. If not a string, it will be
            converted to one using `str()`.

    Returns:
        str: The cleaned text in lowercase, with:
            - Text normalized to Unicode NFKC form.
            - URLs replaced with ' URL '.
            - User mentions (e.g., @username) replaced with ' USER '.
            - '[NAME]' placeholders (case-insensitive) replaced with ' NAME '.
            - HTML entities (&amp;, &lt;, &gt;) unescaped.
            - Excess whitespace collapsed into single spaces and stripped.
    
    Example:
        >>> clean_text("Check out &lt;this&gt; link: http://example.com! @user")
        'check out <this> link:  url !  user'
    """
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)
    text = re.sub(r"@\w+", " USER ", text)
    text = re.sub(r"\[NAME\]", " NAME ", text, flags=re.I)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()



def _svd_n_components(requested: int, n_samples: int, n_features: int) -> int:
    """Determine the valid number of SVD components based on data dimensions and requested value.

    This helper function computes the maximum allowable number of singular value decomposition (SVD)
    components given the shape of the data matrix. The number of components is bounded above by
    the rank of the matrix, which cannot exceed min(n_samples - 1, n_features - 1). It also ensures
    that the result is at least 1 and respects the requested number if within valid bounds.

    Args:
        requested (int): The desired number of components; will be capped to the feasible range.
        n_samples (int): The number of samples (rows) in the data matrix.
        n_features (int): The number of features (columns) in the data matrix.

    Returns:
        int: The adjusted number of SVD components, guaranteed to be:
             - At least 1.
             - At most min(n_features - 1, n_samples - 1).
             - Equal to the requested value if it falls within the valid range.

    Notes:
        - When either n_features or n_samples is 1 or less, the rank of the matrix is at most 1.
        - This function is typically used to prevent invalid SVD computations due to excessive
          component requests relative to data size.

    Example:
        >>> _svd_n_components(10, 100, 50)
        49
        >>> _svd_n_components(5, 100, 50)
        5
        >>> _svd_n_components(5, 1, 1)
        1
    """
    if n_features <= 1 or n_samples <= 1:
        return 1
    cap = min(n_features - 1, n_samples - 1)
    return max(1, min(int(requested), int(cap)))



class TfidfDualSvdConcat(BaseEstimator, TransformerMixin):
    """Text transformer that combines word and character-level TF-IDF features using SVD.

    This transformer applies two independent TF-IDF vectorizations — one at the word level
    and another at the character level (using 'char_wb' analyzer) — followed by dimensionality
    reduction via Truncated SVD. The resulting low-dimensional representations are concatenated
    into a single dense feature vector per document.

    The transformation is robust to varying input sizes, automatically capping the number of
    SVD components based on data dimensions using `_svd_n_components`.

    Parameters
    ----------
    word_max_features : int, default=50_000
        Maximum number of features for the word-level TF-IDF vectorizer.
    char_max_features : int, default=60_000
        Maximum number of features for the character-level TF-IDF vectorizer.
    n_components_word : int, default=180
        Target number of components for word-level SVD. Will be capped by min(n_samples-1, n_features-1).
    n_components_char : int, default=180
        Target number of components for character-level SVD. Will be capped similarly.
    random_state : int, default=SEED
        Random state for reproducibility in SVD.
    max_df_word : float, default=WORD_MAX_DF
        Maximum document frequency for word-level TF-IDF (proportion of documents above which terms are ignored).

    Attributes
    ----------
    word_vec_ : TfidfVectorizer
        Fitted word-level TF-IDF vectorizer.
    char_vec_ : TfidfVectorizer
        Fitted character-level TF-IDF vectorizer.
    svd_w_ : TruncatedSVD
        SVD model trained on word-level TF-IDF output.
    svd_c_ : TruncatedSVD
        SVD model trained on character-level TF-IDF output.

    Methods
    -------
    fit(X, y=None)
        Fit the transformer on the training data by learning TF-IDF and SVD models.
    transform(X)
        Transform the input text data into concatenated SVD-reduced TF-IDF features.

    Notes
    -----
    - The word vectorizer uses unigrams and bigrams, strips accents, and applies sublinear TF scaling.
    - The char vectorizer uses n-grams from 3 to 5 characters within word boundaries (`char_wb`).
    - Both TF-IDF outputs are reduced independently and then horizontally stacked.
    - Output is a float32 array for memory efficiency, suitable for downstream ML models.

    Example
    -------
    >>> transformer = TfidfDualSvdConcat(word_max_features=10000, char_max_features=20000)
    >>> X_text = ["This is a sample.", "Another example sentence."]
    >>> X_transformed = transformer.fit_transform(X_text)
    >>> X_transformed.shape
    (2, 360)  # 180 (word) + 180 (char), assuming no component capping
    """

    def __init__(
        self,
        *,
        word_max_features: int = 50_000,
        char_max_features: int = 60_000,
        n_components_word: int = 180,
        n_components_char: int = 180,
        random_state: int = SEED,
        max_df_word: float = WORD_MAX_DF,
    ):
        self.word_max_features = word_max_features
        self.char_max_features = char_max_features
        self.n_components_word = n_components_word
        self.n_components_char = n_components_char
        self.random_state = random_state
        self.max_df_word = max_df_word

    def fit(self, X, y=None):
        """Fit the transformer on the input text data.

        Learns the word and character-level TF-IDF vectorizers and fits SVD models
        on their outputs to enable later transformation.

        Parameters
        ----------
        X : iterable of str
            Collection of text documents to fit the transformer on.
        y : None
            Ignored, present for API consistency.

        Returns
        -------
        self : TfidfDualSvdConcat
            Fitted transformer instance.
        """
        self.word_vec_ = TfidfVectorizer(
            max_features=self.word_max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=self.max_df_word,
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
        """Transform the input text data into concatenated SVD-reduced features.

        Applies fitted TF-IDF vectorizers and SVD models to produce a dense,
        fixed-length feature vector for each document.

        Parameters
        ----------
        X : iterable of str
            Collection of text documents to transform.

        Returns
        -------
        np.ndarray of shape (n_samples, n_components_word + n_components_char)
            Concatenated word and character-level SVD-transformed features in float32.

        Raises
        ------
        NotFittedError
            If the transformer has not been fitted before calling this method.
        """
        X_w = self.word_vec_.transform(X)
        X_c = self.char_vec_.transform(X)
        z_w = self.svd_w_.transform(X_w)
        z_c = self.svd_c_.transform(X_c)
        return np.hstack([z_w, z_c]).astype(np.float32, copy=False)



def tfidf_word() -> TfidfVectorizer:
    """Create a word-level TF-IDF vectorizer with predefined parameters for text preprocessing.

    This function returns a configured `TfidfVectorizer` instance optimized for word-based
    text representation. It supports unigrams and bigrams, applies sublinear term frequency
    scaling, removes accents, and limits vocabulary size to control memory usage.

    Returns
    -------
    TfidfVectorizer
        A fitted or fittable vectorizer object with the following configuration:
        - `max_features=60_000`: Limits vocabulary to the top 60,000 terms by frequency.
        - `ngram_range=(1, 2)`: Uses both unigrams and bigrams.
        - `min_df=2`: Ignores terms that appear in fewer than 2 documents.
        - `max_df=WORD_MAX_DF`: Ignores terms appearing in more than WORD_MAX_DF proportion of documents.
        - `sublinear_tf=True`: Applies log scaling to term frequencies (i.e., tf → 1 + log(tf)).
        - `strip_accents="unicode"`: Removes accents via Unicode normalization.
        - `dtype=np.float32`: Uses 32-bit floats for memory efficiency.

    Notes
    -----
    The vectorizer must be fitted on a corpus before transforming text data.
    The value of `WORD_MAX_DF` is expected to be defined in the module scope.

    Example
    -------
    >>> vectorizer = tfidf_word()
    >>> X = ["The quick brown fox", "jumps over the lazy dog"]
    >>> X_tfidf = vectorizer.fit_transform(X)
    >>> X_tfidf.shape
    (2, 60000)  # or fewer if vocabulary is smaller

    This vectorizer is suitable for use in pipelines for text classification,
    clustering, or other machine learning tasks requiring dense numerical features.
    """
    return TfidfVectorizer(
        max_features=60_000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=WORD_MAX_DF,
        sublinear_tf=True,
        strip_accents="unicode",
        dtype=np.float32,
    )



def tfidf_word_char_union() -> FeatureUnion:
    """Create a combined word and character-level TF-IDF feature extractor.

    This function returns a `FeatureUnion` object that concatenates features from two
    independent `TfidfVectorizer` instances:
      - A word-level vectorizer using unigrams and bigrams.
      - A character-level vectorizer using n-grams (3–5 characters) within word boundaries.

    The resulting feature set captures both lexical semantics and subword patterns,
    such as prefixes, suffixes, and morphological structures, which can improve performance
    in text classification and other NLP tasks.

    Returns
    -------
    FeatureUnion
        A feature union containing two transformers:
        - 'word': Word-level TF-IDF vectorizer with:
            - `max_features=45_000`
            - `ngram_range=(1, 2)`
            - `min_df=2`, `max_df=WORD_MAX_DF`
            - `sublinear_tf=True`, `strip_accents="unicode"`
            - `dtype=np.float32`
        - 'char': Character-level TF-IDF vectorizer with:
            - `analyzer="char_wb"` (uses word boundaries)
            - `ngram_range=(3, 5)`
            - `max_features=45_000`
            - `min_df=2`, `max_df=0.95`
            - `sublinear_tf=True`
            - `dtype=np.float32`

    Notes
    -----
    - The output is a sparse matrix where columns correspond to concatenated features:
      first from the word vectorizer, then from the char vectorizer.
    - `WORD_MAX_DF` should be defined in the module scope.
    - Both vectorizers are fitted on the same input corpus when `fit` is called.
    - This transformer is suitable for use in scikit-learn pipelines.

    Example
    -------
    >>> union = tfidf_word_char_union()
    >>> X = ["This is a test.", "Another example sentence."]
    >>> X_features = union.fit_transform(X)
    >>> X_features.shape
    (2, 90000)  # ~45k (word) + ~45k (char), depending on vocabulary

    This approach is particularly effective for models that benefit from rich text
    representations, including linear classifiers and shallow neural networks.
    """
    word = TfidfVectorizer(
        max_features=45_000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=WORD_MAX_DF,
        sublinear_tf=True,
        strip_accents="unicode",
        dtype=np.float32,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=45_000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        dtype=np.float32,
    )
    return FeatureUnion([("word", word), ("char", char)])



def _gbdt_clf():
    """Create a gradient boosting classifier with fallback support for missing dependencies.

    This function returns a gradient boosting decision tree (GBDT) classifier configured for
    multiclass classification. It prioritizes LightGBM (`LGBMClassifier`) if available, and
    falls back to scikit-learn's `HistGradientBoostingClassifier` if LightGBM is not installed.

    The returned model includes tuned hyperparameters suitable for robust performance on
    text or tabular data, with regularization and early stopping where applicable.

    Returns
    -------
    Union[lightgbm.LGBMClassifier, HistGradientBoostingClassifier]
        - If `lightgbm` is installed: Returns an `LGBMClassifier` with:
            - `objective="multiclass"`
            - `class_weight="balanced"` to handle imbalanced datasets.
            - `n_estimators=700`, `learning_rate=0.05`, `num_leaves=48`
            - Subsampling and column sampling enabled for robustness.
            - Verbosity silenced via `verbosity=-1`.
        - Otherwise: Returns a `HistGradientBoostingClassifier` with:
            - `learning_rate=0.07`, `max_iter=450`, `max_depth=11`
            - Regularization via `l2_regularization=1.0` and `min_samples_leaf=12`
            - Early stopping enabled using 10% of data as validation set.
            - Binning limited to 255 bins for efficiency.

    Notes
    -----
    - `SEED` must be defined in the module scope for reproducibility.
    - The use of `try/except ImportError` allows optional dependency on LightGBM.
    - Both classifiers support `class_weight="balanced"` to adjust for class imbalance.
    - Recommended for use in pipelines where high predictive accuracy and speed are required.

    Example
    -------
    >>> clf = _gbdt_clf()
    >>> clf.fit(X_train, y_train)
    >>> y_pred = clf.predict(X_test)

    This function enables consistent modeling interface regardless of LightGBM availability.
    """
    try:
        import lightgbm as lgb  # type: ignore

        return lgb.LGBMClassifier(
            objective="multiclass",
            class_weight="balanced",
            n_estimators=700,
            learning_rate=0.05,
            num_leaves=48,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_samples=15,
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



def gbdt_backend() -> str:
    """Determine which gradient boosting backend is available in the current environment.

    This function checks whether the 'lightgbm' library is installed. If it is, returns
    'lightgbm' as the preferred backend; otherwise, falls back to 'hist_gradient_boosting',
    referring to scikit-learn's `HistGradientBoostingClassifier`.

    Returns
    -------
    str
        - 'lightgbm' if the LightGBM package is installed.
        - 'hist_gradient_boosting' if LightGBM is not available.

    Notes
    -----
    - Used to dynamically select or report the active GBDT backend in machine learning pipelines.
    - The import is marked with `# noqa: F401` to suppress unused import warnings,
      as the purpose is only to test availability.
    - Does not instantiate any model — only checks for library presence.

    Example
    -------
    >>> backend = gbdt_backend()
    >>> if backend == "lightgbm":
    ...     print("Using LightGBM for boosted trees.")
    ... else:
    ...     print("Falling back to scikit-learn HistGradientBoosting.")

    This function enables conditional logic in model configuration based on installed dependencies.
    """
    try:
        import lightgbm  # noqa: F401

        return "lightgbm"
    except ImportError:
        return "hist_gradient_boosting"



def make_gbdt_pipeline() -> Pipeline:
    """Create a complete machine learning pipeline for text classification using TF-IDF, SVD, and gradient boosting.

    This function constructs a `Pipeline` that combines text preprocessing and classification:
      1. A custom transformer (`TfidfDualSvdConcat`) extracts and reduces word and character-level TF-IDF features.
      2. A gradient boosting classifier (`_gbdt_clf`) performs multiclass or binary classification,
         with automatic fallback between LightGBM and scikit-learn based on availability.

    Returns
    -------
    sklearn.pipeline.Pipeline
        A pipeline with two stages:
        - 'prep': An instance of `TfidfDualSvdConcat` configured with:
            - `word_max_features=50_000`
            - `char_max_features=60_000`
            - `n_components_word=180`, `n_components_char=180`
            - `random_state=SEED`
        - 'clf': The classifier returned by `_gbdt_clf()`, either:
            - `LGBMClassifier` (if LightGBM is installed)
            - `HistGradientBoostingClassifier` (otherwise)

    Notes
    -----
    - The pipeline encapsulates all steps from raw text to prediction, supporting methods:
        - `.fit(X, y)` — where X is an iterable of strings.
        - `.predict(X)`
        - `.predict_proba(X)`
    - Designed for text classification tasks with high performance and robustness.
    - Uses fixed random state for reproducibility.
    - Requires `SEED` to be defined in the module scope.

    Example
    -------
    >>> pipe = make_gbdt_pipeline()
    >>> X_train = ["This is a positive review.", "I hated this product."]
    >>> y_train = [1, 0]
    >>> pipe.fit(X_train, y_train)
    >>> y_pred = pipe.predict(["Great service!"])
    >>> y_pred
    array([1])

    This pipeline is suitable for production use and cross-validation workflows.
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
            ("clf", _gbdt_clf()),
        ]
    )


def make_learners() -> list[tuple[str, Pipeline | CalibratedClassifierCV]]:
    """Create a list of configured text classification pipelines for comparative modeling.

    This function returns a collection of diverse machine learning pipelines, each tailored
    for text classification tasks using TF-IDF features and different classifiers. The learners
    vary in feature extraction strategies (word-level, word+char-level) and model types,
    enabling benchmarking or ensemble approaches.

    Returns
    -------
    list[tuple[str, Union[Pipeline, CalibratedClassifierCV]]]
        A list of tuples, each containing:
        - A unique identifier string for the learner.
        - A configured estimator (typically a Pipeline or calibrated classifier).

        The returned learners are:
        1. 'sgd_word': SGDClassifier with word-level TF-IDF (unigrams/bigrams).
        2. 'sgd_word_char': SGDClassifier with combined word and character n-gram TF-IDF.
        3. 'complement_nb': Complement Naive Bayes on word-level TF-IDF features.
        4. 'linearsvc_calibrated': Calibrated LinearSVC (with sigmoid calibration) to produce probability estimates.
        5. 'gbdt_tfidf_svd': Gradient boosting pipeline with dual SVD-reduced TF-IDF features.

    Notes
    -----
    - All models use `class_weight="balanced"` where applicable to handle class imbalance.
    - Random states are fixed via `SEED` for reproducibility.
    - Pipelines support raw text input (strings) and can be used directly in `fit()` and `predict()`.
    - The `CalibratedClassifierCV` uses 2-fold cross-validation for probability calibration.
    - Feature vectorizers reuse parameters from `tfidf_word()` where appropriate via `get_params()`.

    Example
    -------
    >>> learners = make_learners()
    >>> for name, model in learners:
    ...     print(f"Fitting {name}...")
    ...     model.fit(X_train, y_train)
    ...     score = model.score(X_test, y_test)
    ...     print(f"{name} accuracy: {score:.3f}")

    This function is ideal for:
      - Model selection and comparison.
      - Ensemble voting or stacking.
      - Benchmarking performance across different algorithm families.

    Requires: SEED must be defined in the module scope.
    """
    out: list = []
    out.append(
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
    out.append(
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
    out.append(
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
    out.append(
        (
            "linearsvc_calibrated",
            CalibratedClassifierCV(base_svc, method="sigmoid", cv=2, n_jobs=-1),
        )
    )
    out.append(("gbdt_tfidf_svd", make_gbdt_pipeline()))
    return out



def align_proba(proba, model_classes, ref_classes) -> np.ndarray:
    """Align predicted class probabilities to a reference class order.

    This function reorders or expands the probability array from a classifier to match
    a specified reference class ordering. It is useful when combining predictions from
    multiple models whose `classes_` attributes may differ in order or coverage.

    Parameters
    ----------
    proba : np.ndarray of shape (n_samples, n_model_classes)
        The predicted probabilities from a classifier, where each column corresponds
        to the probability for a class in `model_classes`.
    model_classes : array-like of shape (n_model_classes,)
        The class labels as ordered in the columns of `proba`. Typically corresponds
        to `model.classes_` from a scikit-learn estimator.
    ref_classes : array-like of shape (n_ref_classes,)
        The desired output class order. The returned probability array will have one
        column per class in `ref_classes`, with probabilities aligned accordingly.

    Returns
    -------
    np.ndarray of shape (n_samples, len(ref_classes))
        Reordered probability array where:
        - Each row sums to <= 1.0 (equal if all classes in `ref_classes` were present in `model_classes`).
        - Missing classes (in `ref_classes` but not in `model_classes`) have zero probability.
        - Column k corresponds to the class `ref_classes[k]`.

    Notes
    -----
    - Class labels are converted to integers via `int(c)` to ensure consistent mapping.
    - If a class in `ref_classes` does not exist in `model_classes`, its probability is set to 0.
    - Preserves `np.float64` precision in output.

    Example
    -------
    >>> proba = np.array([[0.7, 0.3], [0.4, 0.6]])  # classes: [0, 1]
    >>> model_classes = [0, 1]
    >>> ref_classes = [1, 0, 2]
    >>> aligned = align_proba(proba, model_classes, ref_classes)
    >>> aligned
    array([[0.3, 0.7, 0. ],
           [0.6, 0.4, 0. ]])

    This function is essential for ensembling or stacking models with non-uniform class labeling.
    """
    n_samples = proba.shape[0]
    out = np.zeros((n_samples, len(ref_classes)), dtype=np.float64)
    m2j = {int(c): j for j, c in enumerate(model_classes)}
    for k, c in enumerate(ref_classes):
        j = m2j.get(int(c))
        if j is not None:
            out[:, k] = proba[:, j]
    return out


def oof_per_learner(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    learner_specs: list[tuple[str, object]],
) -> tuple[dict[str, np.ndarray], np.ndarray, list[float]]:
    """Generate out-of-fold (OOF) predictions for a set of learners using stratified cross-validation.

    This function performs K-fold stratified cross-validation and collects OOF predicted probabilities
    for each model in `learner_specs`. It also computes macro F1 scores for a blended prediction
    on each fold using configurable weights (`BLEND_WEIGHTS`), enabling model comparison and stacking.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        The input feature matrix. Typically text data transformed into numerical features.
    y : np.ndarray of shape (n_samples,)
        The target labels. Will be converted to integers and sorted to define the reference class order.
    n_splits : int
        Number of folds for StratifiedKFold cross-validation.
    learner_specs : list[tuple[str, object]]
        List of tuples, each containing:
        - A unique name for the learner (str).
        - A scikit-learn estimator instance (e.g., Pipeline) that supports `fit` and `predict_proba`.

    Returns
    -------
    oof : dict[str, np.ndarray]
        Dictionary mapping learner names to their OOF probability arrays of shape (n_samples, n_classes).
        Each row corresponds to the held-out prediction for that sample.
    ref : np.ndarray of shape (n_classes,)
        Sorted array of unique class labels, used as the canonical class ordering.
    fold_scores : list[float]
        List of macro F1 scores (float) for the weighted blend of all models' predictions on each fold.

    Notes
    -----
    - Uses `StratifiedKFold` with `shuffle=True` and fixed `random_state=SEED` for reproducibility.
    - Each model is cloned before fitting to avoid modifying the original instances.
    - Predicted probabilities are aligned to a common class order using `align_proba`.
    - Blending uses weights from a global `BLEND_WEIGHTS` dictionary; defaults to 1.0 if not specified.
    - Final fold predictions are obtained by taking the argmax of the blended probabilities.
    - F1 score is computed with `average="macro"` and `zero_division=0`.

    Example
    -------
    >>> X = np.array([...])  # Preprocessed feature vectors
    >>> y = np.array([0, 1, 2, ...])
    >>> learners = make_learners()  # From make_learners()
    >>> oof_preds, classes, scores = oof_per_learner(X, y, n_splits=5, learner_specs=learners)
    >>> print(f"Mean OOF blend F1: {np.mean(scores):.4f}")

    This function is designed for use in ensemble learning workflows, particularly for generating
    meta-features for stacking or evaluating base learner performance.
    """
    ref = np.sort(np.unique(y.astype(int)))
    n, k = len(y), len(ref)
    names = [n for n, _ in learner_specs]
    oof = {name: np.zeros((n, k), dtype=np.float64) for name in names}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    fold_scores: list[float] = []

    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        fold_blend = np.zeros((len(va_idx), k), dtype=np.float64)
        wsum = 0.0
        for name, est in learner_specs:
            m = clone(est)
            m.fit(X_tr, y_tr)
            p = align_proba(m.predict_proba(X_va), m.classes_, ref)
            oof[name][va_idx] = p
            fold_blend += float(BLEND_WEIGHTS.get(name, 1.0)) * p
            wsum += float(BLEND_WEIGHTS.get(name, 1.0))
        fold_blend /= max(wsum, 1e-9)
        pred = ref[np.argmax(fold_blend, axis=1)]
        fold_scores.append(f1_score(y_va, pred, average="macro", zero_division=0))

    return oof, ref, fold_scores



def weighted_blend_from_oof(oof: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    """Compute a weighted blend of out-of-fold (OOF) predictions from multiple models.

    This function combines predicted probabilities from different learners stored in the `oof`
    dictionary using predefined weights from the global `BLEND_WEIGHTS` dictionary. The result
    is a consensus probability distribution across all classes for each sample.

    Parameters
    ----------
    oof : dict[str, np.ndarray]
        Dictionary mapping model names to their OOF prediction arrays of shape (n_samples, n_classes).
        Each array contains class probabilities produced during cross-validation.
    names : list[str]
        List of learner names (keys in `oof`) to include in the blend. Order does not affect result.

    Returns
    -------
    np.ndarray of shape (n_samples, n_classes)
        The weighted average of OOF probabilities, normalized by the sum of weights.
        - Each row sums to 1.0 (valid probability distribution).
        - If a name is not in `BLEND_WEIGHTS`, defaults to weight = 1.0.
        - Uses `np.float64` precision for numerical stability.

    Notes
    -----
    - The number of classes `k` and number of samples are inferred from the first model's OOF array.
    - All OOF arrays are assumed to be aligned to the same class order (e.g., via `align_proba`).
    - A small epsilon (`1e-9`) prevents division by zero if total weight is zero.
    - Designed to work with output from `oof_per_learner`.

    Example
    -------
    >>> oof_preds, _, _ = oof_per_learner(X, y, n_splits=5, learner_specs=learners)
    >>> blended = weighted_blend_from_oof(oof_preds, names=["sgd_word", "gbdt_tfidf_svd"])
    >>> blended.shape
    (n_samples, n_classes)

    This function is useful for creating ensemble predictions or preparing meta-features in stacking models.
    """
    k = next(iter(oof.values())).shape[1]
    acc = np.zeros((next(iter(oof.values())).shape[0], k), dtype=np.float64)
    wsum = 0.0
    for name in names:
        w = float(BLEND_WEIGHTS.get(name, 1.0))
        acc += w * oof[name]
        wsum += w
    return acc / max(wsum, 1e-9)



def stack_features(oof: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    """Construct a feature matrix for meta-learning by horizontally stacking OOF predictions.

    This function creates a unified input dataset for a meta-model (stacker) by concatenating
    the out-of-fold (OOF) predicted probabilities from specified base learners. Each model's
    probability output becomes additional features for the next level of learning.

    Parameters
    ----------
    oof : dict[str, np.ndarray]
        Dictionary mapping model names to their OOF prediction arrays of shape (n_samples, n_classes).
        These arrays are typically generated via cross-validation (e.g., using `oof_per_learner`).
    names : list[str]
        List of learner names (keys in `oof`) whose predictions should be included in the stack.
        Order determines column order in the output.

    Returns
    -------
    np.ndarray of shape (n_samples, n_classes * len(names))
        A dense feature matrix where:
        - Each row corresponds to a sample.
        - Columns are the concatenated predicted class probabilities from each named model.
        - Output is in `float64` precision and avoids unnecessary copying when possible.

    Notes
    -----
    - All OOF arrays must have the same number of samples and be aligned to the same class order.
    - The resulting matrix has `len(names) * n_classes` columns.
    - Suitable for use as input to train a meta-classifier in a stacking ensemble.
    - Uses `np.hstack` followed by `astype(np.float64, copy=False)` for memory efficiency.

    Example
    -------
    >>> oof_preds, _, _ = oof_per_learner(X, y, n_splits=5, learner_specs=learners)
    >>> X_stack = stack_features(oof_preds, names=["sgd_word", "complement_nb"])
    >>> X_stack.shape
    (n_samples, 2 * n_classes)

    This function enables second-level generalization by allowing a meta-model to learn
    how to best combine base learner predictions.
    """
    return np.hstack([oof[n] for n in names]).astype(np.float64, copy=False)


def apply_log_bias(proba: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Apply additive log-space bias to predicted probabilities and return class predictions.

    This function adjusts class probability estimates by adding a bias term in log-space,
    then selects the most likely class. It is commonly used to correct for label imbalance
    or incorporate prior knowledge into model predictions.

    Parameters
    ----------
    proba : np.ndarray of shape (n_samples, n_classes)
        Predicted class probabilities from a classifier, where each row sums to approximately 1.
    bias : np.ndarray of shape (n_classes,)
        Bias vector to be added in log-space. Typically derived from log of class priors
        or tuned on validation data.

    Returns
    -------
    np.ndarray of shape (n_samples,)
        Predicted class labels after applying log bias, obtained via `argmax` over classes.

    Notes
    -----
    - Input probabilities are clipped to [1e-12, 1.0] to avoid numerical issues with `log`.
    - The bias is broadcast across all samples using `None` (i.e., `bias[None, :]`).
    - Final prediction is based on `np.argmax(log(p) + bias)`, which is equivalent to
      scaling the original probabilities by `exp(bias)` and taking the maximum.
    - Does not re-normalize the output; it only changes the decision rule.

    Example
    -------
    >>> proba = np.array([[0.7, 0.3], [0.4, 0.6]])
    >>> bias = np.array([0.1, -0.1])  # Favor class 0
    >>> pred = apply_log_bias(proba, bias)
    >>> pred
    array([0, 0])

    This method is useful in post-processing calibrated or uncalibrated model outputs
    when misclassification costs or class distributions differ between training and deployment.
    """
    p = np.clip(proba, 1e-12, 1.0)
    return np.argmax(np.log(p) + bias[None, :], axis=1)



def tune_class_bias(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_iter: int,
    step: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Hill-climb per-class additive bias in log-probability space for macro-F1.

    This function optimizes a class-specific bias vector that is added to the log-probabilities
    of predicted class distributions to maximize the macro-averaged F1 score. It performs a
    stochastic hill-climbing search over the bias values, adjusting one class at a time.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground truth labels.
    proba : np.ndarray of shape (n_samples, n_classes)
        Predicted class probabilities from a model (e.g., OOF predictions).
    n_iter : int
        Number of iterations to run the hill-climbing algorithm.
    step : float
        Step size for bias updates; each trial adjusts one class bias by ±step.
    rng : np.random.Generator
        Random number generator for reproducible sampling of class indices and step directions.

    Returns
    -------
    best_b : np.ndarray of shape (n_classes,)
        The optimized bias vector that maximizes macro F1 when applied via `apply_log_bias`.
        Initialized as zeros and updated during search.
    best_f1 : float
        The achieved macro F1 score on `y_true` using the final bias-adjusted predictions.

    Notes
    -----
    - Bias is applied in log-space: `pred = argmax(log(proba) + bias)`.
    - At each iteration, a random class is selected and its bias is adjusted by +step or -step.
    - Update is accepted only if it improves the macro F1 score.
    - Useful for correcting model miscalibration due to label imbalance or domain shift.
    - Optimization is greedy and may converge to a local optimum.

    Example
    -------
    >>> y_true = np.array([0, 1, 2, 0, 1])
    >>> proba = np.random.dirichlet([1]*3, size=5).astype(np.float32)  # mock probabilities
    >>> rng = np.random.default_rng(42)
    >>> bias, score = tune_class_bias(y_true, proba, n_iter=100, step=0.1, rng=rng)
    >>> print(f"Optimized bias: {bias}, Macro F1: {score:.4f}")

    This method is suitable for post-processing ensemble or single-model predictions
    in classification tasks where balanced performance across classes is critical.
    """
    n_c = proba.shape[1]
    best_b = np.zeros(n_c, dtype=np.float32)
    best_f1 = f1_score(y_true, apply_log_bias(proba, best_b), average="macro", zero_division=0)
    for _ in range(n_iter):
        c = int(rng.integers(0, n_c))
        delta = float(rng.choice([-step, step]))
        trial = best_b.copy()
        trial[c] += delta
        pred_idx = apply_log_bias(proba, trial)
        sc = f1_score(y_true, pred_idx, average="macro", zero_division=0)
        if sc > best_f1:
            best_f1 = sc
            best_b = trial.astype(np.float32)
    return best_b, best_f1



def main() -> None:
    """Execute the final ensemble and calibration pipeline for text classification.

    This function orchestrates a complete machine learning workflow including:
      - Loading and preprocessing training and test data.
      - Generating out-of-fold (OOF) predictions using multiple base learners.
      - Blending and stacking models with optional meta-learning.
      - Tuning class-specific log-probability bias to maximize macro F1.
      - Refitting all models on full training data.
      - Producing calibrated predictions on the test set.
      - Saving submission file and metadata.

    Command-line arguments:
        --no-meta : bool
            If set, disables the meta logistic regression model on stacked OOF features.
        --bias-iters : int, default=6000
            Number of iterations for hill-climbing class bias optimization.
        --bias-step : float, default=0.012
            Step size used in bias tuning.
        --quick-bias : bool
            If set, overrides `--bias-iters` to 1500 for faster execution.

    Workflow Steps
    --------------
    1. Parse arguments and initialize RNG with fixed SEED.
    2. Load train/test CSV files and apply text cleaning.
    3. Encode labels with LabelEncoder.
    4. Determine number of CV folds based on train/test size ratio.
    5. Generate OOF predictions via `oof_per_learner` for all learners from `make_learners`.
    6. Compute weighted blend of OOF predictions using `BLEND_WEIGHTS`.
    7. Optionally train a logistic regression meta-model on stacked OOF features.
    8. Select best probability source (blend or meta) for bias tuning.
    9. Optimize per-class log-space bias using `tune_class_bias` to maximize OOF macro F1.
    10. Refit all base learners on full training set.
    11. Predict on test data and combine using same blending scheme.
    12. Apply tuned bias to test probabilities and map back to original labels.
    13. Ensure submission matches sample ID order.
    14. Save submission CSV and detailed metadata JSON.

    Outputs
    -------
    - Submission file: `{OUT_SUBMISSIONS}/final_submission.csv`
        Contains columns: `id`, `target`
    - Metadata file: `{OUT_META}/final_pipeline.json`
        Includes performance metrics, configuration, timestamps, and model info.

    Prints
    ------
    - Pipeline configuration summary.
    - OOF performance at each stage (blend, meta, after bias).
    - Per-model fitting times.
    - Final output paths and wall time.

    Notes
    -----
    - Uses stratified OOF evaluation with macro F1 as primary metric.
    - Supports reproducibility via fixed `SEED`.
    - Handles case where test IDs are in different order than sample submission.
    - Probability clipping prevents numerical issues in log-space operations.
    - The meta-model uses L2-regularized logistic regression (`C=1.5`) with balanced class weights.

    Example
    -------
    $ python pipeline.py --bias-iters 3000 --no-meta
    === Final pipeline ===
    Train=12345  Test=5432  OOF folds=3  GBDT=lightgbm
    Preprocess: NFKC + URL/USER/NAME, lowercased, max_df_word=0.95
    OOF weighted blend macro-F1: 0.7421 ...
    ...

    Saved: submissions/final_submission.csv
    Meta:  meta/final_pipeline.json
    Wall:  124.3s

    This script is intended as the final inference step in a classification pipeline,
    suitable for generating production-ready submissions.
    """
    ap = argparse.ArgumentParser(description="Final ensemble + calibration pipeline.")
    ap.add_argument("--no-meta", action="store_true", help="Disable meta logistic on stacked OOF probs.")
    ap.add_argument("--bias-iters", type=int, default=6000)
    ap.add_argument("--bias-step", type=float, default=0.012)
    ap.add_argument("--quick-bias", action="store_true", help="Use 1500 bias iterations.")
    args = ap.parse_args()
    if args.quick_bias:
        args.bias_iters = 1500

    rng = np.random.default_rng(SEED)
    t0 = time.perf_counter()

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")

    X = train_df["text"].fillna("").map(clean_text).to_numpy()
    X_test = test_df["text"].fillna("").map(clean_text).to_numpy()
    le = LabelEncoder()
    y = le.fit_transform(train_df["target"].values)

    n_test = len(test_df)
    n_splits = max(2, int(round(len(train_df) / n_test)))
    specs = make_learners()
    names = [n for n, _ in specs]

    print("=== Final pipeline ===")
    print(f"Train={len(train_df)}  Test={n_test}  OOF folds={n_splits}  GBDT={gbdt_backend()}")
    print(f"Preprocess: NFKC + URL/USER/NAME, lowercased, max_df_word={WORD_MAX_DF}")

    oof, ref_arr, fold_f1 = oof_per_learner(X, y, n_splits, specs)
    p_blend = weighted_blend_from_oof(oof, names)
    blend_f1 = f1_score(y, ref_arr[np.argmax(p_blend, axis=1)], average="macro", zero_division=0)
    print(f"OOF weighted blend macro-F1: {blend_f1:.4f}  | per-fold: {[round(x, 4) for x in fold_f1]}")

    use_meta = not args.no_meta
    p_for_bias = p_blend
    meta_f1 = -1.0
    meta_model = None
    if use_meta:
        X_st = stack_features(oof, names)
        meta_model = LogisticRegression(
            C=1.5,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=4000,
            random_state=SEED,
        )
        meta_model.fit(X_st, y)
        p_meta = meta_model.predict_proba(X_st).astype(np.float64)
        meta_f1 = f1_score(y, np.argmax(p_meta, axis=1), average="macro", zero_division=0)
        print(f"OOF meta-stack macro-F1:     {meta_f1:.4f}")
        if meta_f1 > blend_f1:
            p_for_bias = p_meta
            print("→ Using meta-stack probabilities for bias tuning & test.")
        else:
            print("→ Meta did not beat blend on OOF; keeping weighted blend.")
            meta_model = None

    bias, bias_f1 = tune_class_bias(y, p_for_bias, args.bias_iters, args.bias_step, rng)
    print(f"OOF after log-bias tune:       {bias_f1:.4f}")

    print("Refitting all learners on full data…")
    wsum = sum(BLEND_WEIGHTS.get(n, 1.0) for n in names)
    test_acc = np.zeros((len(X_test), len(ref_arr)), dtype=np.float64)
    test_blocks: dict[str, np.ndarray] = {}
    for name, est in specs:
        t1 = time.perf_counter()
        est.fit(X, y)
        p = align_proba(est.predict_proba(X_test), est.classes_, ref_arr)
        test_blocks[name] = p
        test_acc += (BLEND_WEIGHTS.get(name, 1.0) / wsum) * p
        print(f"  {name}: {time.perf_counter() - t1:.1f}s")

    if meta_model is not None:
        X_te_st = np.hstack([test_blocks[n] for n in names]).astype(np.float64, copy=False)
        p_test = meta_model.predict_proba(X_te_st).astype(np.float64)
    else:
        p_test = test_acc

    p_test = np.clip(p_test, 1e-12, 1.0)
    pred_idx = apply_log_bias(p_test, bias)
    pred_labels = le.inverse_transform(pred_idx)

    sub = pd.DataFrame({"id": test_df["id"], "target": pred_labels.astype(int)})
    if list(sample["id"]) != list(sub["id"]):
        m = dict(zip(test_df["id"].values, pred_labels.astype(int), strict=True))
        sub = sample[["id"]].copy()
        sub["target"] = sub["id"].map(m).astype(int)

    OUT_SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    OUT_META.mkdir(parents=True, exist_ok=True)
    sub_path = OUT_SUBMISSIONS / "final_submission.csv"
    sub[["id", "target"]].to_csv(sub_path, index=False)

    meta_json = {
        "oof_blend_macro_f1": float(blend_f1),
        "oof_meta_macro_f1": float(meta_f1) if use_meta else None,
        "used_meta_stack": meta_model is not None,
        "oof_after_bias_macro_f1": float(bias_f1),
        "cv_n_splits": n_splits,
        "blend_weights": BLEND_WEIGHTS,
        "gbdt_backend": gbdt_backend(),
        "learners": names,
        "bias_iters": args.bias_iters,
        "submission": str(sub_path),
        "seconds": round(time.perf_counter() - t0, 2),
    }
    with open(OUT_META / "final_pipeline.json", "w", encoding="utf-8") as f:
        json.dump(meta_json, f, indent=2)

    print(f"\nSaved: {sub_path}")
    print(f"Meta:  {OUT_META / 'final_pipeline.json'}")
    print(f"Wall:  {meta_json['seconds']:.1f}s")


if __name__ == "__main__":
    main()
