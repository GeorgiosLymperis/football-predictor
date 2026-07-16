"""MLflow tracking/registry helpers: log a run, decide whether to promote
it, export a promoted model's lean artifacts for serving.
"""
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import mlflow
import yaml
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

REPO_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = REPO_ROOT / 'models'
CHAMPION_ALIAS = 'champion'


def configure() -> None:
    mlflow.set_tracking_uri(f'sqlite:///{REPO_ROOT / "mlflow.db"}')
    mlflow.set_registry_uri(f'sqlite:///{REPO_ROOT / "mlflow.db"}')


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'unknown'


def log_run(model_name: str, params: dict, metrics: dict, tags: dict,
            python_model, artifacts: dict, input_example, signature) -> str:
    """Log one training run and register it as a new version of `model_name`.

    Returns the new model version.
    """
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags({**tags, 'git_sha': git_sha()})
        model_info = mlflow.pyfunc.log_model(
            name='model',
            python_model=python_model,
            artifacts=artifacts,
            input_example=input_example,
            signature=signature,
            registered_model_name=model_name,
            code_paths=[str(REPO_ROOT / 'src' / 'match_predict')],
        )
        version = getattr(model_info, 'registered_model_version', None)
        if version is None:
            latest = MlflowClient().get_latest_versions(model_name)
            version = max(v.version for v in latest if v.run_id == run.info.run_id)
        return str(version)


def get_champion_rps(model_name: str) -> Optional[float]:
    """RPS the currently-aliased champion was gated on, or None if there
    isn't one yet (first-ever run for this model)."""
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
    except MlflowException:
        return None
    run = client.get_run(mv.run_id)
    return run.data.metrics.get('rps')


def promote_if_better(model_name: str, version: str, candidate_rps: float,
                       baseline_rps: float, margin: float = 0.0) -> bool:
    """Alias `version` as champion iff it beats the baseline and (if one
    exists) beats the current champion by at least `margin` relative RPS.
    """
    if candidate_rps >= baseline_rps:
        return False

    champion_rps = get_champion_rps(model_name)
    if champion_rps is not None and candidate_rps >= champion_rps * (1 - margin):
        return False

    MlflowClient().set_registered_model_alias(model_name, CHAMPION_ALIAS, version)
    return True


def export_champion(model_name: str, version: str, rps: float, baseline_rps: float,
                     artifact_files: dict[str, str], out_dir: Path,
                     market_rps: Optional[float] = None, market_n_matches: Optional[int] = None) -> Path:
    """Copy a promoted version's lean artifacts into `out_dir` and write
    metadata.yaml. The contents of this directory (not the MLflow registry) is what
    streamlit_app.py actually reads at serve time.

    Args:
        artifact_files: {destination filename -> source path on the local
            filesystem}, e.g. the paths passed as `artifacts=` to log_run().
        out_dir: destination directory, e.g. `models/<league>/<model_name>/`.
        market_rps, market_n_matches: the market-implied-odds benchmark
            over the same holdout, when available.
    """
    client = MlflowClient()
    mv = client.get_model_version(model_name, version)
    out_dir.mkdir(parents=True, exist_ok=True)

    for dest_name, src_path in artifact_files.items():
        shutil.copyfile(src_path, out_dir / dest_name)

    metadata = {
        'model_name': model_name,
        'version': version,
        'run_id': mv.run_id,
        'rps': rps,
        'baseline_rps': baseline_rps,
        'market_rps': market_rps,
        'market_n_matches': market_n_matches,
        'promoted_at': datetime.now(timezone.utc).isoformat(),
        'git_sha': git_sha(),
    }
    with open(out_dir / 'metadata.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(metadata, f, sort_keys=False)
    return out_dir
