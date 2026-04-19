import os
import json

MODEL_NAME = "gemma3:4b"
EMBEDDING_MODEL = "embeddinggemma"
SUPPORTED_FORMATS = ["pdf", "txt", "docx", "html", "md", "png", "jpg", "jpeg"]

# LLM provider: "ollama" or "openrouter"
LLM_PROVIDER = "ollama"
OPENROUTER_API_KEY = ""
OPENROUTER_MODEL = "openrouter/elephant-alpha"

# Бесплатные модели OpenRouter (без ограничений, суффикс :free)
OPENROUTER_FREE_MODELS = [
    # Проверено как рабочие (апрель 2026)
    "openrouter/elephant-alpha",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "openai/gpt-oss-120b:free",
    "openai/gpt-oss-20b:free",
    "z-ai/glm-4.5-air:free",
    "google/gemma-3-27b-it:free",
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "nvidia/nemotron-nano-9b-v2:free",
    "google/gemma-4-26b-a4b-it:free",
    "google/gemma-4-31b-it:free",
]

_SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache", "settings.json")


def load_settings() -> dict:
    import logging as _log
    try:
        with open(_SETTINGS_FILE, "r", encoding="utf-8") as f:
            s = json.load(f)
        _log.info(f"[CONFIG] Настройки загружены из {_SETTINGS_FILE}: provider={s.get('provider')}, model={s.get('openrouter_model') or s.get('ollama_model')}")
        return s
    except FileNotFoundError:
        _log.info(f"[CONFIG] Файл настроек не найден ({_SETTINGS_FILE}), используются дефолты")
        return {}
    except Exception as e:
        _log.warning(f"[CONFIG] Ошибка загрузки настроек: {e}")
        return {}


def save_settings(settings: dict):
    import logging as _log
    try:
        os.makedirs(os.path.dirname(_SETTINGS_FILE), exist_ok=True)
        with open(_SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
        _log.info(f"[CONFIG] Настройки сохранены в {_SETTINGS_FILE}: {settings}")
    except Exception as e:
        _log.error(f"[CONFIG] Не удалось сохранить настройки: {e}")
        print(f"[CONFIG] Не удалось сохранить настройки: {e}")


def get_llm_settings() -> dict:
    """Возвращает актуальные настройки LLM (из файла или дефолты)."""
    s = load_settings()
    return {
        "provider":         s.get("provider", LLM_PROVIDER),
        "ollama_model":     s.get("ollama_model", MODEL_NAME),
        "openrouter_key":   s.get("openrouter_key", OPENROUTER_API_KEY),
        "openrouter_model": s.get("openrouter_model", OPENROUTER_MODEL),
    }
