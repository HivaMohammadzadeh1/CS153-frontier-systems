from learning_memory_os.config import Settings


def test_settings_reads_from_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql://u:p@localhost:5433/db"
    )
    monkeypatch.setenv("LMOS_LOG_DIR", "/tmp/logs")
    monkeypatch.setenv("LMOS_DEFAULT_TOKEN_BUDGET", "4000")

    s = Settings()
    assert s.anthropic_api_key == "sk-ant-test"
    assert s.openai_api_key == "sk-test"
    assert s.default_token_budget == 4000
