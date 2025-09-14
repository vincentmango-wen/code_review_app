"""
utility.py
app.py から切り出したユーティリティ関数をまとめたモジュール。
初心者向けに分かりやすく書かれています。

関数:
- save_uploaded_file(uploaded_file, target_dir)
- extract_zip(zip_path, extract_to)
- run_subprocess(cmd, cwd, timeout)
- is_zip_file(uploaded_file)

このファイルを作ることで `app.py` が読みやすくなり、ユーティリティの再利用も容易になります。
"""

import zipfile
import tempfile
import time
import os
from pathlib import Path
from typing import Any, Dict

from logger import get_logger
from config import settings

logger = get_logger(__name__)


def save_uploaded_file(uploaded_file: Any, target_dir: Path) -> Path:
    """
    アップロードされたファイルを保存する簡単な関数。
    uploaded_file: Streamlit が受け取ったファイルオブジェクト
    target_dir: 保存先のディレクトリパス
    戻り値: 保存したファイルのパス
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / uploaded_file.name
    try:
        with open(dest, "wb") as f:
            f.write(uploaded_file.getbuffer())
        logger.info("Saved uploaded file %s to %s", uploaded_file.name, dest)
        return dest
    except Exception:
        logger.exception("Failed to save uploaded file %s", getattr(uploaded_file, "name", "<unknown>"))
        raise


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """
    ZIP ファイルを展開（解凍）して指定ディレクトリにファイルを取り出します。
    zip_path: ZIP ファイルのパス
    extract_to: 解凍先のディレクトリ
    """
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_to)
        logger.info("Extracted zip %s to %s", zip_path, extract_to)
    except zipfile.BadZipFile:
        logger.exception("Bad zip file: %s", zip_path)
        raise


def run_subprocess(cmd, cwd: Path, timeout: int) -> Dict[str, Any]:
    """
    外部コマンド（flake8, pytest など）を実行して結果を取得する関数。
    cmd: 実行するコマンドのリスト（例: ["flake8", "."]）
    cwd: コマンドを実行するカレントディレクトリ
    timeout: タイムアウト（秒）
    戻り値: 辞書 {
        "returncode": プロセスの終了コード（0 は成功）、
        "stdout": 標準出力のテキスト、
        "stderr": 標準エラーのテキスト、
        "elapsed": 実行にかかった秒数
    }
    """
    t0 = time.time()
    try:
        # subprocess.run を遅延 import（軽量化目的）
        import subprocess

        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        elapsed = time.time() - t0
        result = {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "elapsed": round(elapsed, 2),
        }
        logger.info("Ran subprocess %s in %s: rc=%s", cmd, cwd, result["returncode"])
        return result
    except subprocess.TimeoutExpired as e:
        elapsed = time.time() - t0
        logger.warning("Subprocess %s timed out after %s seconds", cmd, timeout)
        return {
            "returncode": -1,
            "stdout": e.stdout or "",
            "stderr": f"Timeout after {timeout} seconds",
            "elapsed": round(elapsed, 2),
        }


def is_zip_file(uploaded_file: Any) -> bool:
    # ファイル名が .zip で終わっているかを確認する簡易判定
    name = getattr(uploaded_file, "name", "").lower()
    return name.endswith(".zip")


def safe_parse_json(s: str) -> Dict[str, Any]:
    """Try to parse JSON and return empty dict on failure (logs exception)."""
    import json

    try:
        return json.loads(s)
    except json.JSONDecodeError:
        logger.exception("JSON decode failed")
        return {}
