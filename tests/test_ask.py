from types import SimpleNamespace

from app import ask as ask_module


def test_ask_displays_input_and_output_token_counts(monkeypatch, capsys):
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(prompt_token_count=12, candidates_token_count=34),
        text="回答",
    )

    def generate_content(**kwargs):
        assert kwargs == {"model": ask_module.MODEL, "contents": "質問"}
        return response

    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=generate_content)
    )
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(ask_module.genai, "Client", lambda **_: client)

    assert ask_module.ask("質問") == "回答"
    assert "[トークン] 入力: 12, 出力: 34" in capsys.readouterr().out


def test_ask_displays_unknown_when_usage_metadata_is_missing(monkeypatch, capsys):
    response = SimpleNamespace(usage_metadata=None, text="回答")

    def generate_content(**kwargs):
        assert kwargs == {"model": ask_module.MODEL, "contents": "質問"}
        return response

    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=generate_content)
    )
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(ask_module.genai, "Client", lambda **_: client)

    assert ask_module.ask("質問") == "回答"
    assert "[トークン] 入力: 不明, 出力: 不明" in capsys.readouterr().out


def test_chat_keeps_history_and_reset_starts_a_new_conversation(monkeypatch, capsys):
    conversations = []

    class FakeChat:
        def __init__(self):
            self.messages = []

        def send_message(self, message):
            self.messages.append(message)
            return SimpleNamespace(
                usage_metadata=SimpleNamespace(
                    prompt_token_count=10, candidates_token_count=20
                ),
                text=f"{message}への回答",
            )

    def create_chat(*, model):
        assert model == ask_module.MODEL
        conversation = FakeChat()
        conversations.append(conversation)
        return conversation

    client = SimpleNamespace(chats=SimpleNamespace(create=create_chat))
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(ask_module.genai, "Client", lambda **_: client)
    inputs = iter(["最初の質問", "続きの質問", "/reset", "新しい会話", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    ask_module.chat()

    assert len(conversations) == 2
    assert conversations[0].messages == ["最初の質問", "続きの質問"]
    assert conversations[1].messages == ["新しい会話"]
    output = capsys.readouterr().out
    assert "[案内] 会話をリセットしました。" in output
    assert "[回答]\n続きの質問への回答" in output


def test_main_starts_chat_when_question_is_omitted(monkeypatch):
    started = []
    monkeypatch.setattr("sys.argv", ["app.ask"])
    monkeypatch.setattr(ask_module, "chat", lambda: started.append(True))

    ask_module.main()

    assert started == [True]


def test_main_uses_single_question_mode_when_question_is_given(monkeypatch, capsys):
    asked = []
    monkeypatch.setattr("sys.argv", ["app.ask", "質問"])
    monkeypatch.setattr(
        ask_module, "ask", lambda question: asked.append(question) or "回答"
    )

    ask_module.main()

    assert asked == ["質問"]
    assert "[質問] 質問" in capsys.readouterr().out
