"""
streamlit_app.py
最小限の Code Review Bot フロントエンド（アップロード → 解析結果表示）のスケルトン。

このファイルは初心者向けにコメントを追加しています。
主な役割:
- ユーザーがファイル（.py または .zip）をアップロードできる簡単な UI を提供します。
- アップロードされたコードに対して flake8（静的解析）と pytest（テスト）を実行します。
- 実行結果を表示し、必要に応じて LLM（言語モデル）にレビューを依頼するためのコンテキストを作ります。

動かし方:
1. 仮想環境を作る（任意）
2. 必要なパッケージをインストール: pip install streamlit flake8 pytest
3. streamlit run app.py でアプリを起動

注意: このサンプルはローカル実験向けです。本番では未検証のコードの実行はコンテナなどで隔離してください。
"""

import streamlit as st
import zipfile
import tempfile
import os
import shutil
from pathlib import Path
import time
import json
from dotenv import load_dotenv

# ユーティリティ関数を外部モジュールに切り出しました
import utility as ut

# ------------- Config -------------
LINTER_CMD = ["flake8", "--max-line-length=120"]  # flake8 を呼ぶコマンド（行長は調整可）
PYTEST_CMD = ["pytest", "-q", "--disable-warnings"]  # pytest を呼ぶコマンド
LINTER_TIMEOUT = 30  # Linter の最大実行時間（秒）
PYTEST_TIMEOUT = 60  # Pytest の最大実行時間（秒）
MAX_UPLOAD_SIZE_MB = 10  # アップロード許容サイズ（MB、ソフトリミット）


# ------------- Streamlit UI -------------
# .env ファイルがあれば読み込み（任意）
load_dotenv()
# 環境変数から API キーを取得（UI から上書き可能）
openai_api_key = os.getenv("OPENAI_API_KEY")

# Streamlit アプリのページ設定（タイトルやレイアウト）
st.set_page_config(page_title="コードレビュー君 - MVP", layout="centered")

st.title("コードレビュー君 — MVP (Upload → Linter / Test → Results)")
st.info(
    """まずはサイドバーで OpenAI API Key を設定してください。\n
    アップロードした Python ファイルまたはプロジェクト ZIP に対して、flake8 と pytest を実行し、その結果を表示します。\n
    後続で LLM に渡す想定の出力が確認できます。"""
)

with st.sidebar:
    st.header("API Key入力")
    # Allow user to provide OpenAI API key at runtime (helps Streamlit processes pick it up)
    api_key_input = st.text_input("OpenAI API Key", type="password")
    if api_key_input:
        # サイドバーで入力されたキーを環境変数にセットする（同一プロセス内で有効）
        os.environ["OPENAI_API_KEY"] = api_key_input
        st.success("OPENAI API key は設定されました。")

    linter_cmd_str = st.text_input("Linter command", " ".join(LINTER_CMD))
    pytest_cmd_str = st.text_input("Pytest command", " ".join(PYTEST_CMD))
    st.caption("ツールがローカルにインストールされている必要があります。")

uploaded = st.file_uploader("Upload a Python file or a project ZIP", type=["py", "zip"])

if uploaded is None:
    # ファイルがまだアップロードされていない場合は案内して処理を中断
    st.info("ファイルをアップロードしてください（.py または .zip）。")
    st.stop()

# Soft size check
uploaded_size_mb = len(uploaded.getbuffer()) / (1024 * 1024)
if uploaded_size_mb > MAX_UPLOAD_SIZE_MB:
    st.warning(f"アップロードサイズが大きめです： {uploaded_size_mb:.1f} MB（上限 {MAX_UPLOAD_SIZE_MB} MB）")

analyze = st.button("Analyze now")  # ユーザーがクリックすると解析処理が始まる

if analyze:
    # 一時作業ディレクトリを作る（ここにアップロードファイルを保存して解析する）
    scratch = Path(tempfile.mkdtemp(prefix="code-review-"))
    st.success(f"作業用ディレクトリを作成しました: `{scratch}`")
    try:
        # アップロードされたファイルを保存し、プロジェクトルートを決める
        saved = ut.save_uploaded_file(uploaded, scratch)
        project_root = scratch

        # ZIP の場合は解凍してプロジェクトルートを探す
        if ut.is_zip_file(uploaded):
            st.info("ZIP を検出 → 展開します...")
            extract_dir = scratch / "extracted"
            extract_dir.mkdir()
            ut.extract_zip(saved, extract_dir)
            # もし解凍結果のトップレベルに1つだけフォルダがあれば、それをプロジェクトルートとする
            entries = list(extract_dir.iterdir())
            if len(entries) == 1 and entries[0].is_dir():
                project_root = entries[0]
            else:
                project_root = extract_dir
            st.write(f"プロジェクトルートを `{project_root}` に設定しました")
        else:
            # single .py file; create a small project structure so pytest can run (if any tests exist)
            project_root = scratch

        # flake8 を実行してコードスタイルの問題をチェックする
        st.info("Linter を実行しています...")
        linter_cmd = linter_cmd_str.split()
        linter_res = ut.run_subprocess(linter_cmd + ["."], cwd=project_root, timeout=LINTER_TIMEOUT)

        # pytest を実行してテスト結果を取得する
        st.info("pytest を実行しています（テストがある場合）...")
        pytest_cmd = pytest_cmd_str.split()
        pytest_res = ut.run_subprocess(pytest_cmd, cwd=project_root, timeout=PYTEST_TIMEOUT)

        # flake8 の標準出力を表示（エラーがあれば stderr も表示）
        with st.expander("Linter (flake8) 結果", expanded=False):
            st.code(linter_res["stdout"] or "(no output)")
            if linter_res["stderr"]:
                st.error("Linter stderr:")
                st.code(linter_res["stderr"])

        # pytest の出力（テスト結果）を表示
        with st.expander("Pytest 結果", expanded=False):
            st.code(pytest_res["stdout"] or "(no tests / no output)")
            if pytest_res["stderr"]:
                st.error("Pytest stderr:")
                st.code(pytest_res["stderr"])

        # LLM（言語モデル）に渡すためのコンテキストを作成する
        # ここには linter や pytest の結果やプロジェクトルートなどを入れる
        context = {
            "project_root": str(project_root),
            "uploaded_name": uploaded.name,
            "linter": {"returncode": linter_res["returncode"], "stdout": linter_res["stdout"]},
            "pytest": {"returncode": pytest_res["returncode"], "stdout": pytest_res["stdout"]},
            # 実運用ではファイルの中身を添付することもあるが、サイズに注意
        }

        with st.expander("LLM に渡す準備済みのコンテキスト", expanded=False):
            # context を見やすく表示（開発時の確認用）
            st.json(context, width=200)
        # llm_review モジュールの遅延読み込み（初回表示を速くするため）
        try:
            from llm_review import run_review  # 必要になったときに読み込む
        except Exception as e:
            st.error(f"LangChain モジュールの読み込みに失敗しました: {e}")
            st.stop()

        # LLM に渡すためのファイルを集める（多すぎると遅くなるので制限）
        st.info("プロジェクトファイルを収集しています（プロンプト用に縮約）...")
        file_dict = {}
        max_files = 12  # limit breadth for speed
        max_bytes = 80 * 1024  # limit per file size
        try:
            gathered = 0
            for root, dirs, files in os.walk(project_root):
                # 隠しフォルダや仮想環境フォルダは除外
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ("__pycache__", ".venv", "venv", "env")]
                for fname in files:
                    # 読み込む拡張子を制限して速度を確保
                    if not fname.lower().endswith((".py", ".md", ".txt")):
                        continue
                    fpath = Path(root) / fname
                    try:
                        if fpath.stat().st_size > max_bytes:
                            # ファイルが大きすぎたらスキップ
                            continue
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as rf:
                            file_dict[str(fpath.relative_to(project_root))] = rf.read()
                            gathered += 1
                            if gathered >= max_files:
                                break
                    except Exception:
                        # 読めないファイルは飛ばす
                        continue
                if gathered >= max_files:
                    break
        except Exception as e:
            st.warning(f"ファイル収集中に問題が発生しました: {e}")
        if not file_dict:
            # LLM に渡すファイルが無ければレビューできないので警告
            st.warning("LLM に渡すファイルが見つかりませんでした。最低1つの .py/.md/.txt ファイルが必要です。")
        else:
            st.success(f"{len(file_dict)} 件のファイルを収集しました。LLM にレビューを依頼します…")
            # 実際に LLM にレビューを依頼
            with st.spinner("LangChain 経由で LLM を呼び出し中…"):
                try:
                    review = run_review(context, file_dict)
                except Exception as e:
                    # ここでエラーが出るときは API キーがない、ネットワークの問題、または LLM の設定不備の可能性
                    st.error(f"LLM 呼び出し / 解析でエラーが発生しました: {e}")
                    review = None
            if review is not None:
                st.subheader("LLM レビュー要約")
                st.write(review.summary)
                st.subheader("LLM 指摘一覧")
                if not review.issues:
                    # 指摘が空の場合は問題無し
                    st.info("指摘はありませんでした。")
                else:
                    # 指摘一覧を表示（やさしい形式）
                    for issue in review.issues:
                        st.markdown(
                            f"- [**{issue.severity.upper()}**] {issue.category} — `{issue.file}`:{issue.start_line}-{issue.end_line}  ")
                        st.markdown(f"  ID: `{issue.id}`  ")
                        st.markdown(f"  説明: {issue.explanation}")
                        st.markdown(f"  提案: {issue.suggestion}")
                        # もし差分（patch）があるなら表示
                        if issue.patch and issue.patch.replacement:
                            st.markdown(
                                f"  パッチ: {issue.patch.type} {issue.patch.start_line}-{issue.patch.end_line}")
                            st.code(issue.patch.replacement, language="diff")
                st.success("LLM レビュー完了")

    except Exception as e:
        st.error(f"Error during analysis: {e}")
