"""第1回: 素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 対話モード

対話モードでは会話履歴を引き継ぎ、/reset でリセットできる。
ツールを持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 演習で使う講座固有の質問。正しい答えは glossary の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def print_token_usage(response: genai.types.GenerateContentResponse) -> None:
    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count if usage is not None else None
    output_tokens = usage.candidates_token_count if usage is not None else None
    print(
        "[トークン] "
        f"入力: {input_tokens if input_tokens is not None else '不明'}, "
        f"出力: {output_tokens if output_tokens is not None else '不明'}"
    )


def ask(question: str) -> str:
    client = genai.Client(api_key=get_api_key())
    response = client.models.generate_content(model=MODEL, contents=question)
    print_token_usage(response)

    return response.text


def chat() -> None:
    client = genai.Client(api_key=get_api_key())
    conversation = client.chats.create(model=MODEL)
    print("[案内] /reset で会話をリセット、/exit で終了します。")

    while True:
        try:
            question = input("あなた> ").strip()
        except EOFError:
            print()
            return

        if not question:
            continue
        if question in {"/exit", "exit", "quit"}:
            return
        if question == "/reset":
            conversation = client.chats.create(model=MODEL)
            print("[案内] 会話をリセットしました。")
            continue

        try:
            response = conversation.send_message(question)
        except errors.APIError as e:
            print(api_error_message(e.code, e.message))
            continue

        print_token_usage(response)
        print(f"[回答]\n{response.text}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?")
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    try:
        if args.question is None:
            chat()
        else:
            print(f"[質問] {args.question}\n")
            print(ask(args.question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
