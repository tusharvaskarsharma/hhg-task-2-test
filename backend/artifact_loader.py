import os
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, Optional, Any

import pandas as pd
import onnxruntime as ort
import hnswlib

from backend.config import settings

logger = logging.getLogger(__name__)


class LanguageArtifacts:
    """Loaded retrieval artifacts for one language."""

    def __init__(self, language: str):
        self.language = language

        self.manifest_data: Dict[str, Any] = {}
        self.metadata_df: Optional[pd.DataFrame] = None
        self.hnsw_index: Optional[hnswlib.Index] = None
        self.bm25_obj: Any = None

        self.status = {
            "manifest": False,
            "bm25": False,
            "hnsw": False,
            "metadata": False,
            "valid": False,
        }

        self.errors = []


class ArtifactLoader:
    """
    Production artifact loader.

    Supports:
        HI + BN + EN

    Shared:
        multilingual-e5-small ONNX model

    Per-language:
        metadata
        HNSW
        BM25
        manifest/contract
    """

    SUPPORTED_LANGUAGES = ("hi", "bn", "en")
    EXPECTED_MODEL = "intfloat/multilingual-e5-small"
    EXPECTED_DIMENSION = 384
    EXPECTED_HNSW_SPACE = "cosine"

    def __init__(self):
        self.status = {
            "valid": False,
            "languages": {},
            "onnx": False,
            "validation_summary": False,
        }

        self.errors = []

        # language -> LanguageArtifacts
        self.languages: Dict[str, LanguageArtifacts] = {}

        # Shared ONNX model
        self.onnx_session: Optional[ort.InferenceSession] = None

        # Backward-compatible aliases.
        self.manifest_data = {}
        self.metadata_df = None
        self.hnsw_index = None
        self.bm25_obj = None

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def initialize(self):
        """
        Load the complete production artifact bundle.

        Raises no exception to the caller.
        Check:
            loader_instance.status["valid"]
            loader_instance.errors
        """

        self.errors = []

        try:
            artifact_root = self._artifact_root()

            if not artifact_root.is_dir():
                raise ValueError(
                    f"Artifact root not found: {artifact_root}"
                )

            logger.info(
                "Initializing HHG production artifacts from %s",
                artifact_root,
            )

            # ----------------------------------------------------------
            # 1. Shared ONNX model
            # ----------------------------------------------------------
            self._load_onnx()

            # ----------------------------------------------------------
            # 2. Per-language retrieval artifacts
            # ----------------------------------------------------------
            for language in self.SUPPORTED_LANGUAGES:
                logger.info(
                    "Loading production artifacts for language=%s",
                    language,
                )

                artifacts = LanguageArtifacts(language)

                try:
                    self._load_language(language, artifacts)
                    artifacts.status["valid"] = True

                except Exception as exc:
                    artifacts.errors.append(str(exc))
                    artifacts.status["valid"] = False

                    self.errors.append(
                        f"{language}: {exc}"
                    )

                    logger.exception(
                        "Failed loading language=%s",
                        language,
                    )

                self.languages[language] = artifacts

                self.status["languages"][language] = dict(
                    artifacts.status
                )

            # ----------------------------------------------------------
            # 3. Validation summary
            # ----------------------------------------------------------
            self._check_validation_summary()

            # ----------------------------------------------------------
            # 4. Final validation
            # ----------------------------------------------------------
            failed_languages = [
                language
                for language, artifacts in self.languages.items()
                if not artifacts.status["valid"]
            ]

            if not self.status["onnx"]:
                self.errors.append(
                    "Shared ONNX model failed to load."
                )

            if failed_languages:
                self.errors.append(
                    "Failed language artifacts: "
                    + ", ".join(failed_languages)
                )

            self.status["valid"] = (
                self.status["onnx"]
                and not failed_languages
                and len(self.errors) == 0
            )

            # Backward-compatible aliases.
            self._update_legacy_aliases()

            if self.status["valid"]:
                logger.info(
                    "HHG artifact initialization successful: "
                    "HI + BN + EN"
                )
            else:
                logger.error(
                    "HHG artifact initialization failed: %s",
                    self.errors,
                )

        except Exception as exc:
            self.errors.append(str(exc))
            self.status["valid"] = False

            logger.exception(
                "Artifact initialization failed"
            )

    # ------------------------------------------------------------------
    # PATH HELPERS
    # ------------------------------------------------------------------

    def _artifact_root(self) -> Path:
        """
        Resolve artifact root from backend settings.
        """

        root = getattr(
            settings,
            "HHG_ARTIFACT_DIR",
            None,
        )

        if not root:
            raise ValueError(
                "settings.HHG_ARTIFACT_DIR is not configured."
            )

        return Path(root).expanduser().resolve()

    def _language_root(
        self,
        language: str,
    ) -> Path:
        """
        Resolve language-specific artifact directory.

        Supports both:

            artifacts/hi/
            artifacts/bn/
            artifacts/en/

        and the older flat layout:

            artifacts/hnsw/
            artifacts/bm25/
            artifacts/metadata/
        """

        root = self._artifact_root()

        candidate = root / language

        if candidate.is_dir():
            return candidate

        # Fallback to flat structure.
        return root

    # ------------------------------------------------------------------
    # LANGUAGE LOADING
    # ------------------------------------------------------------------

    def _load_language(
        self,
        language: str,
        artifacts: LanguageArtifacts,
    ):
        language_root = self._language_root(language)

        logger.info(
            "Language root for %s: %s",
            language,
            language_root,
        )

        self._load_language_manifest(
            language,
            language_root,
            artifacts,
        )

        self._load_language_metadata(
            language,
            language_root,
            artifacts,
        )

        self._load_language_hnsw(
            language,
            language_root,
            artifacts,
        )

        self._load_language_bm25(
            language,
            language_root,
            artifacts,
        )

    # ------------------------------------------------------------------
    # MANIFEST
    # ------------------------------------------------------------------

    def _find_manifest(
        self,
        language_root: Path,
    ) -> Optional[Path]:

        candidates = [
            language_root / "build_manifest.json",
            language_root / "manifest.json",
            language_root / "config.json",
        ]

        for path in candidates:
            if path.is_file():
                return path

        # Global fallback.
        root = self._artifact_root()

        candidates = [
            root / "build_manifest.json",
            root / "manifest.json",
            root / "config.json",
        ]

        for path in candidates:
            if path.is_file():
                return path

        return None

    def _load_language_manifest(
        self,
        language: str,
        language_root: Path,
        artifacts: LanguageArtifacts,
    ):

        manifest_path = self._find_manifest(
            language_root
        )

        if manifest_path is None:
            raise ValueError(
                f"Manifest not found for language={language}"
            )

        with open(
            manifest_path,
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        artifacts.manifest_data = data

        # --------------------------------------------------------------
        # Flexible field lookup.
        # --------------------------------------------------------------

        model = (
            data.get("embedding_model")
            or data.get("model")
        )

        dimension = (
            data.get("embedding_dimension")
            or data.get("dimension")
        )

        normalized = (
            data.get("normalized_embeddings")
            if "normalized_embeddings" in data
            else data.get("normalize_embeddings")
        )

        hnsw_space = data.get("hnsw_space")

        # --------------------------------------------------------------
        # Model
        # --------------------------------------------------------------

        if model and model != self.EXPECTED_MODEL:
            raise ValueError(
                f"{language}: embedding model mismatch. "
                f"Expected {self.EXPECTED_MODEL}, got {model}"
            )

        # --------------------------------------------------------------
        # Dimension
        # --------------------------------------------------------------

        if dimension is not None:
            if int(dimension) != self.EXPECTED_DIMENSION:
                raise ValueError(
                    f"{language}: embedding dimension mismatch. "
                    f"Expected {self.EXPECTED_DIMENSION}, "
                    f"got {dimension}"
                )

        # --------------------------------------------------------------
        # Normalization
        # --------------------------------------------------------------

        if normalized is not None:
            if bool(normalized) is not True:
                raise ValueError(
                    f"{language}: normalized embeddings "
                    f"are required."
                )

        # --------------------------------------------------------------
        # HNSW space
        # --------------------------------------------------------------

        if hnsw_space:
            if hnsw_space != self.EXPECTED_HNSW_SPACE:
                raise ValueError(
                    f"{language}: HNSW space mismatch. "
                    f"Expected {self.EXPECTED_HNSW_SPACE}, "
                    f"got {hnsw_space}"
                )

        artifacts.status["manifest"] = True

        logger.info(
            "Manifest validated for %s",
            language,
        )

    # ------------------------------------------------------------------
    # METADATA
    # ------------------------------------------------------------------

    def _find_metadata(
        self,
        language_root: Path,
    ) -> Optional[Path]:

        candidates = [
            language_root / "metadata" / "metadata.parquet",
            language_root / "metadata" / "data.parquet",
            language_root / "metadata.parquet",
        ]

        for path in candidates:
            if path.is_file():
                return path

        # Search metadata directory.
        metadata_dir = language_root / "metadata"

        if metadata_dir.is_dir():
            parquet_files = sorted(
                metadata_dir.glob("*.parquet")
            )

            if parquet_files:
                return parquet_files[0]

        return None

    def _load_language_metadata(
        self,
        language: str,
        language_root: Path,
        artifacts: LanguageArtifacts,
    ):
        if getattr(self, "shared_metadata_df", None) is not None:
            df = self.shared_metadata_df
            language_col = self._first_existing(df.columns, ("language", "lang"))
            id_col = self._first_existing(df.columns, ("passage_id", "id", "doc_id", "chunk_id"))
            if language_col:
                artifacts.metadata_df = df[df[language_col] == language].reset_index(drop=True).set_index(id_col, drop=False)
            else:
                artifacts.metadata_df = df.set_index(id_col, drop=False)
            artifacts.status["metadata"] = True
            logger.info("Shared metadata reused for language=%s", language)
            return

        metadata_path = self._find_metadata(
            language_root
        )

        if metadata_path is None:
            raise ValueError(
                f"Metadata parquet not found for "
                f"language={language}"
            )

        import pyarrow.parquet as pq
        schema = pq.read_schema(metadata_path)
        all_cols = schema.names
        
        id_col = self._first_existing(all_cols, ("passage_id", "id", "doc_id", "chunk_id"))
        text_col = self._first_existing(all_cols, ("text", "passage", "content", "english_text"))
        language_col = self._first_existing(all_cols, ("language", "lang"))

        if id_col is None:
            raise ValueError(f"{language}: metadata has no ID column.")
        if text_col is None:
            raise ValueError(f"{language}: metadata has no text column.")

        cols_to_load = [id_col, text_col]
        if language_col:
            cols_to_load.append(language_col)

        # Optimize memory usage by loading only required columns
        df = pd.read_parquet(
            metadata_path,
            columns=cols_to_load
        )

        if df.empty:
            raise ValueError(
                f"{language}: metadata is empty."
            )

        logger.info(
            "Metadata loaded: language=%s rows=%d id=%s text=%s",
            language,
            len(df),
            id_col,
            text_col,
        )

        if language_col:
            logger.info(
                "Metadata language column=%s",
                language_col,
            )

        self.shared_metadata_df = df
        if language_col:
            artifacts.metadata_df = df[df[language_col] == language].reset_index(drop=True).set_index(id_col, drop=False)
        else:
            artifacts.metadata_df = df.set_index(id_col, drop=False)
        artifacts.status["metadata"] = True

    # ------------------------------------------------------------------
    # HNSW
    # ------------------------------------------------------------------

    def _find_hnsw(
        self,
        language_root: Path,
        language: str,
    ) -> Optional[Path]:

        candidates = [
            language_root / "hnsw" / language / "index.bin",
            language_root / "hnsw" / "index.bin",
            language_root / "hnsw" / "hnsw_index.bin",
            language_root / "index.bin",
            language_root / "hnsw_index.bin",
        ]

        for path in candidates:
            if path.is_file():
                return path

        hnsw_dir = language_root / "hnsw" / language

        if hnsw_dir.is_dir():
            binaries = sorted(
                hnsw_dir.glob("*.bin")
            )

            if binaries:
                return binaries[0]

        return None

    def _load_language_hnsw(
        self,
        language: str,
        language_root: Path,
        artifacts: LanguageArtifacts,
    ):

        index_path = self._find_hnsw(
            language_root,
            language
        )

        if index_path is None:
            raise ValueError(
                f"HNSW index not found for "
                f"language={language}"
            )

        index = hnswlib.Index(
            space=self.EXPECTED_HNSW_SPACE,
            dim=self.EXPECTED_DIMENSION,
        )

        index.load_index(
            str(index_path)
        )

        index.set_ef(
            int(
                getattr(
                    settings,
                    "HNSW_EF_SEARCH",
                    64,
                )
            )
        )

        artifacts.hnsw_index = index
        artifacts.status["hnsw"] = True

        logger.info(
            "HNSW loaded: language=%s count=%d",
            language,
            index.get_current_count(),
        )

    # ------------------------------------------------------------------
    # BM25
    # ------------------------------------------------------------------

    def _find_bm25(
        self,
        language_root: Path,
        language: str,
    ) -> Optional[Path]:

        bm25_dir = language_root / "bm25" / language

        if not bm25_dir.is_dir():
            bm25_dir = language_root / "bm25"

        if not bm25_dir.is_dir():
            return None

        # Prefer explicit pickle artifacts.
        preferred = [
            "bm25.pkl",
            "bm25_index.pkl",
            "index.pkl",
            "bm25.bin",
            "index.bin",
            "bm25.json",
            "index.json",
        ]

        for filename in preferred:
            path = bm25_dir / filename

            if path.is_file():
                return path

        # Fallback: deterministic first file.
        candidates = sorted(
            [
                *bm25_dir.glob("*.pkl"),
                *bm25_dir.glob("*.pickle"),
                *bm25_dir.glob("*.bin"),
                *bm25_dir.glob("*.json"),
            ]
        )

        if candidates:
            return candidates[0]

        return None

    def _load_language_bm25(
        self,
        language: str,
        language_root: Path,
        artifacts: LanguageArtifacts,
    ):

        bm25_path = self._find_bm25(
            language_root,
            language
        )

        # Check for bm25s layout
        bm25_lang_dir = language_root / "bm25" / language
        if bm25_lang_dir.is_dir() and (bm25_lang_dir / "data.csc.index.npy").is_file():
            import bm25s
            artifacts.bm25_obj = bm25s.BM25.load(str(bm25_lang_dir), load_corpus=False)
            artifacts.status["bm25"] = True
            logger.info("BM25s loaded: language=%s dir=%s", language, bm25_lang_dir)
            return

        if bm25_path is None:
            raise ValueError(
                f"BM25 artifact not found for "
                f"language={language}"
            )

        suffix = bm25_path.suffix.lower()

        if suffix in (".pkl", ".pickle", ".bin"):

            with open(
                bm25_path,
                "rb",
            ) as f:
                obj = pickle.load(f)

            artifacts.bm25_obj = obj

        elif suffix == ".json":

            with open(
                bm25_path,
                "r",
                encoding="utf-8",
            ) as f:
                artifacts.bm25_obj = json.load(f)

        else:
            raise ValueError(
                f"Unsupported BM25 format: {bm25_path}"
            )

        artifacts.status["bm25"] = True

        logger.info(
            "BM25 loaded: language=%s file=%s",
            language,
            bm25_path,
        )

    # ------------------------------------------------------------------
    # ONNX
    # ------------------------------------------------------------------

    def _find_onnx(self) -> Optional[Path]:

        root = self._artifact_root()

        candidates = [
            root / "embedding" / "onnx" / "model.onnx",
            root / "embedding" / "onnx" / "model_quint8_avx2.onnx",
            root / "onnx" / "model.onnx",
            root / "onnx" / "model_quint8_avx2.onnx",
        ]

        for path in candidates:
            if path.is_file():
                return path

        # Generic fallback.
        for path in root.rglob("*.onnx"):
            return path

        return None

    def _load_onnx(self):

        onnx_path = self._find_onnx()

        if onnx_path is None:
            raise ValueError(
                "ONNX model not found in artifact bundle."
            )

        logger.info(
            "Loading ONNX model: %s",
            onnx_path,
        )

        sess_options = ort.SessionOptions()
        if settings.ONNX_INTRA_THREADS > 0:
            sess_options.intra_op_num_threads = settings.ONNX_INTRA_THREADS
        if settings.ONNX_INTER_THREADS > 0:
            sess_options.inter_op_num_threads = settings.ONNX_INTER_THREADS

        self.onnx_session = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_options,
            providers=[
                "CPUExecutionProvider"
            ],
        )

        inputs = self.onnx_session.get_inputs()
        outputs = self.onnx_session.get_outputs()

        if not inputs:
            raise ValueError(
                "ONNX model has no inputs."
            )

        if not outputs:
            raise ValueError(
                "ONNX model has no outputs."
            )

        output_shape = outputs[0].shape

        # Some ONNX graphs expose dynamic dimensions.
        last_dim = output_shape[-1]

        if isinstance(last_dim, int):
            if last_dim != self.EXPECTED_DIMENSION:
                raise ValueError(
                    "ONNX output dimension mismatch. "
                    f"Expected {self.EXPECTED_DIMENSION}, "
                    f"got {last_dim}"
                )

        logger.info(
            "ONNX loaded successfully: "
            "inputs=%s outputs=%s",
            [x.name for x in inputs],
            [x.name for x in outputs],
        )

        self.status["onnx"] = True

    # ------------------------------------------------------------------
    # VALIDATION SUMMARY
    # ------------------------------------------------------------------

    def _check_validation_summary(self):

        root = self._artifact_root()

        candidates = [
            root / "validation_report.json",
            root / "validation_summary.json",
            root / "evaluation_report.json",
        ]

        found = False

        for path in candidates:
            if path.is_file():
                found = True

                try:
                    with open(
                        path,
                        "r",
                        encoding="utf-8",
                    ) as f:
                        report = json.load(f)

                    if isinstance(report, dict):
                        passed = report.get(
                            "passed",
                            True,
                        )

                        if passed is False:
                            raise ValueError(
                                f"Artifact validation report "
                                f"failed: {path}"
                            )

                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid validation JSON: {path}"
                    ) from exc

                break

        self.status[
            "validation_summary"
        ] = found

        if found:
            logger.info(
                "Artifact validation summary found."
            )
        else:
            logger.warning(
                "No validation summary/report found."
            )

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _first_existing(
        columns,
        candidates,
    ) -> Optional[str]:

        column_set = set(columns)

        for candidate in candidates:
            if candidate in column_set:
                return candidate

        return None

    def _update_legacy_aliases(self):

        # Preserve compatibility with existing code that expects
        # loader_instance.metadata_df, hnsw_index, bm25_obj.

        primary = self.languages.get("hi")

        if primary is not None:
            self.manifest_data = (
                primary.manifest_data
            )

            self.metadata_df = (
                primary.metadata_df
            )

            self.hnsw_index = (
                primary.hnsw_index
            )

            self.bm25_obj = (
                primary.bm25_obj
            )

    # ------------------------------------------------------------------
    # RETRIEVAL ACCESS
    # ------------------------------------------------------------------

    def get_language(
        self,
        language: str,
    ) -> LanguageArtifacts:

        language = language.lower().strip()

        if language not in self.languages:
            raise ValueError(
                f"Unsupported language: {language}. "
                f"Supported: {self.SUPPORTED_LANGUAGES}"
            )

        artifacts = self.languages[language]

        if not artifacts.status["valid"]:
            raise RuntimeError(
                f"Artifacts for {language} are not valid: "
                f"{artifacts.errors}"
            )

        return artifacts

    def get_hnsw(
        self,
        language: str,
    ):
        return self.get_language(
            language
        ).hnsw_index

    def get_bm25(
        self,
        language: str,
    ):
        return self.get_language(
            language
        ).bm25_obj

    def get_metadata(
        self,
        language: str,
    ):
        return self.get_language(
            language
        ).metadata_df

    def get_onnx_session(self):
        if not self.status["onnx"]:
            raise RuntimeError(
                "ONNX model is not initialized."
            )

        return self.onnx_session

    def get_status(self):
        return {
            **self.status,
            "errors": list(self.errors),
        }


# ----------------------------------------------------------------------
# SINGLETON
# ----------------------------------------------------------------------

loader_instance = ArtifactLoader()