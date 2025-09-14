"""
llm_review.py
Code Review Bot のための LangChain + Pydantic OutputParser テンプレート。

- LLM からの期待される JSON 出力に対応する Pydantic スキーマを定義します。
- パーサーからのフォーマット指示を含む PromptTemplate を構築します。
- LLMChain の使用例と PydanticOutputParser による安全な JSON 解析を示します。

注意:
- model_name / LLM クラスは環境に合わせて調整してください（gpt-4o-mini、gpt-4o など）。
- 使用する langchain バージョンが PydanticOutputParser をサポートしていることを確認してください（ほとんどの最新版は対応しています）。
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain, PromptTemplate
from langchain.output_parsers import PydanticOutputParser
import os
import json

# ----------------------------
# 1) Pydantic スキーマ定義（データの形を決める）
# ここでは LLM から返ってくる JSON の構造を事前に定義します。
# Pydantic を使うと、受け取った JSON が期待する形かどうかを簡単に検査できます。
# ----------------------------

class Patch(BaseModel):
    # Patch は「どの行の差分をどう変えるか」を表します。
    # type: replace/insert/delete のいずれかを想定
    type: str = Field(..., description="パッチ種別: replace/insert/delete")
    # 差し替え開始行と終了行（行番号）
    start_line: int
    end_line: int
    # 差し替えるテキスト本体
    replacement: str


class Issue(BaseModel):
    # Issue は LLM が報告する「問題・指摘」を表します。
    id: str
    file: str
    start_line: int
    end_line: int
    # 重要度（low/medium/high/critical のような文字列を想定）
    severity: str  # e.g., low/medium/high/critical
    # カテゴリ（style, bug, security, test など）
    category: str  # e.g., style, bug, security, test
    # なぜ問題かを説明する短い文章
    explanation: str
    # どのように直すかの提案（短く）
    suggestion: str
    # もし可能なら差分（Patch）を添付する
    patch: Optional[Patch] = None


class ReviewOutput(BaseModel):
    # ReviewOutput は LLM が返す全体の結果の形です。
    issues: List[Issue]
    # 全体の簡単な要約
    summary: str


# ----------------------------
# ----------------------------
# 2) PydanticOutputParser の作成
# LLM の出力を JSON として受け取り、それを Pydantic モデル（ReviewOutput）に
# 安全にパース（変換）するための準備をします。
# ----------------------------
parser = PydanticOutputParser(pydantic_object=ReviewOutput)

# ----------------------------
# 3) Prompt の定義
# LLM に渡す「指示文」をテンプレートとして定義します。
# system プロンプトはアシスタントの役割や出力フォーマットのルールを示します。
SYSTEM_PROMPT = """
あなたは慎重で保守的なコードレビューアシスタントです。次のものが与えられます：
1) 簡単な linter / テストのレポート、
2) 1つ以上の「ファイルスニペット」（ファイルパス + ファイル内容）、
これらを受けて、提供された JSON スキーマの指示に正確に従う JSON オブジェクトを出力してください。

重要なルール:
- 出力は有効な JSON であり、スキーマに厳密に従うこと。
- JSON 以外の散文（説明文）を出力しないこと。
- 問題が見つからない場合は、"issues": [] と適切な "summary" を返してください。
- 可能であれば各指摘に対してテキスト差分（patch）を含めてください（行番号を使用）。
- 提案は最小限かつ具体的にしてください。
- 提案は日本語で書いてください。
"""

PROMPT_TEMPLATE = """{system_instructions}

CONTEXT:
{context}

FILES:
{files}

INSTRUCTIONS:
{format_instructions}

Begin.
"""


prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["system_instructions", "context", "files", "format_instructions"],
)

# ----------------------------
# 4) LLM / Chain setup
# ----------------------------
# The LLM and chain are created at runtime inside run_review() so that
# environment variables (OPENAI_API_KEY or OPEN_API_KEY) set by the caller
# (for example via the Streamlit sidebar) are picked up.

# ----------------------------
# 5) Helper: build files string
# ----------------------------
def build_files_block(file_dict: dict) -> str:
    """
    file_dict: {"path/to/file.py": "file content", ...}
    Prompt に含めるためのコンパクトなファイルブロックを返します（大きい場合は切り詰めます）。
    - 目的: LLM に渡すファイル一覧のテキスト化
    - 大きすぎるファイルは途中で切ることでプロンプト長を抑えます
    """
    parts = []
    MAX_LINES = 400  # prompt が大きくなりすぎないように上限行数を設定
    for path, content in file_dict.items():
        lines = content.splitlines()
        if len(lines) > MAX_LINES:
            # 長すぎるファイルは先頭 400 行だけ使う
            excerpt = "\n".join(lines[:MAX_LINES]) + "\n...<TRUNCATED>\n"
        else:
            excerpt = content
        parts.append(f"--- FILE: {path} ---\n{excerpt}\n")
    return "\n".join(parts)


# ----------------------------
# 6) Example runner
# ----------------------------
def run_review(context: dict, file_dict: dict) -> ReviewOutput:
    """
    context: linter/pytest の出力を含む辞書（文字列化されていても可）
    file_dict: パス -> ファイル内容 のマッピング（文字列）
    戻り値: 解析された ReviewOutput（Pydantic モデル）
    """
    # まず、ファイルのブロックを作り、Pydantic パーサーからフォーマット指示を取得します。
    files_block = build_files_block(file_dict)
    format_instructions = parser.get_format_instructions()
    prompt_input = {
        "system_instructions": SYSTEM_PROMPT.strip(),
        "context": json.dumps(context, indent=2, ensure_ascii=False),
        "files": files_block,
        "format_instructions": format_instructions,
    }

    # Ensure API key is available (Streamlit may set OPENAI_API_KEY at runtime)
    # Streamlit などから環境変数で API キーが設定されていることを確認
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPEN_API_KEY")
    if not api_key:
        # 初心者向けメッセージ: API キーが設定されていないと LLM を呼べません
        raise EnvironmentError(
            "OpenAI の API キーが見つかりません。環境変数 OPENAI_API_KEY（または OPEN_API_KEY） を設定するか、アプリの UI から入力してください。"
        )

    # LLM とチェインを実行時に作ることで、上で確認した環境変数を確実に使えます。
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)
    chain = LLMChain(llm=llm, prompt=prompt)

    # 1) LLM にプロンプトを送り、生の応答文字列を受け取る
    response = chain.run(**prompt_input)

    # 2) Parse via PydanticOutputParser
    # 2) PydanticOutputParser で安全にパース（期待した JSON 形式か検査）
    try:
        parsed: ReviewOutput = parser.parse(response)
        return parsed
    except Exception as e:
        # もしパーサーで失敗したら、応答文字列から JSON 部分を抜き出して手動でパースを試みる
        import re
        m = re.search(r"(\{[\s\S]*\})", response)
        if m:
            raw_json = m.group(1)
            try:
                data = json.loads(raw_json)
                parsed = ReviewOutput.parse_obj(data)
                return parsed
            except Exception as e2:
                # さらに失敗したら、解析エラーとして詳しい情報を返す
                raise ValueError(f"LLM 出力の解析に失敗しました: {e2}\nRaw: {raw_json}") from e2
        # JSON 部分も見つからなければ全体の生データを含めて例外を投げる
        raise ValueError(f"LLM 出力の解析に失敗しました。エラー: {e}\nRaw LLM output:\n{response}") from e

# ----------------------------
# 7) Small example / test
# ----------------------------
if __name__ == "__main__":
    # Example context simulating streamlit_app output
    sample_context = {
        "project_root": "/tmp/code-review-abc",
        "uploaded_name": "example_project.zip",
        "linter": {
            "returncode": 1,
            "stdout": "example.py:3:1: F401 'os' imported but unused\nexample.py:10:5: E225 missing whitespace around operator\n"
        },
        "pytest": {"returncode": 1, "stdout": "FAILED tests/test_example.py::test_sum - AssertionError: x != y\n"},
    }

    # Example file contents (very small)
    file_contents = {
        "example.py": "import os\n\ndef add(a,b):\n    return a+b\n",
        "tests/test_example.py": "from example import add\n\ndef test_sum():\n    assert add(1,2) == 4\n",
    }

    print("LLM にレビューを依頼します（OpenAI API を呼び出します）...")
    result = run_review(sample_context, file_contents)
    print("解析結果（Pydantic）:")
    print(result.json(indent=2, ensure_ascii=False))
