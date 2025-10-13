import os

def create_structure():
    # Получаем текущую директорию, где находится скрипт
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Список папок для создания
    directories = [
        os.path.join(root_dir, '.github/workflows'),
        os.path.join(root_dir, 'src/ui'),
        os.path.join(root_dir, 'tests'),
        os.path.join(root_dir, 'docs/images'),
        os.path.join(root_dir, 'assets/test_files'),
        os.path.join(root_dir, 'assets/icons'),
        os.path.join(root_dir, 'scripts')
    ]

    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)

    # Список файлов с базовым содержимым
    files_content = {
        os.path.join(root_dir, 'src/__init__.py'): '',
        os.path.join(root_dir, 'src/main.py'): '# Точка входа приложения\nif __name__ == "__main__":\n    print("RAG Assistant started")\n',
        os.path.join(root_dir, 'src/config.py'): '# Конфигурация проекта\nOLLAMA_MODEL = "llama3"\nEMBEDDING_MODEL = "all-MiniLM-L6-v2"\nSUPPORTED_FORMATS = [".pdf", ".txt", ".docx", ".html", ".md", ".png", ".jpg"]\n',
        os.path.join(root_dir, 'src/indexer.py'): '# Модуль индексации\n# Извлечение текста, создание эмбеддингов, сохранение в FAISS\npass\n',
        os.path.join(root_dir, 'src/rag.py'): '# Модуль RAG\n# Поиск и генерация ответов с Ollama\npass\n',
        os.path.join(root_dir, 'src/ui/__init__.py'): '',
        os.path.join(root_dir, 'src/ui/main_window.py'): '# Главное окно UI (PyQt5)\n# Чат, drag-and-drop\npass\n',
        os.path.join(root_dir, 'src/ui/tray_icon.py'): '# Трей-иконка (Pystray)\npass\n',
        os.path.join(root_dir, 'src/security.py'): '# Модуль безопасности\n# Проверки доступа, логи в SQLite\npass\n',
        os.path.join(root_dir, 'tests/__init__.py'): '',
        os.path.join(root_dir, 'tests/test_indexer.py'): '# Тесты для indexer\n# Проверка парсинга и эмбеддингов\npass\n',
        os.path.join(root_dir, 'tests/test_rag.py'): '# Тесты для RAG\n# Проверка генерации ответов\npass\n',
        os.path.join(root_dir, 'tests/test_ui.py'): '# Тесты для UI\n# Требуется pytest-qt\npass\n',
        os.path.join(root_dir, 'docs/mkdocs.yml'): 'site_name: RAG Assistant\nnav:\n  - Home: index.md\n  - API: api.md\ntheme: material\n',
        os.path.join(root_dir, 'docs/index.md'): '# Добро пожаловать в документацию RAG Assistant\n\nЛокальный ассистент для поиска и анализа файлов.\n',
        os.path.join(root_dir, 'docs/api.md'): '# API документация\n\nГенерируется из docstrings модулей src/.\n',
        os.path.join(root_dir, 'scripts/setup.py'): '# Скрипт установки\n# poetry install\npass\n',
        os.path.join(root_dir, 'scripts/build.py'): '# Скрипт сборки\n# pyinstaller --onefile src/main.py\npass\n',
        os.path.join(root_dir, 'scripts/run_tests.py'): '# Запуск тестов\n# pytest tests/ --cov=src/\npass\n',
        os.path.join(root_dir, '.gitignore'): '# Gitignore для Python\n__pycache__/\n*.pyc\n*.pyo\n*.pyd\n.Python\nbuild/\ndist/\n*.egg-info/\n*.egg\n\n# Виртуальные окружения\nvenv/\n.env\n\n# FAISS и кэши\nfaiss_index/\nmetadata.db\n\n# Логи и временные файлы\nlogs/\n*.tmp\n\n# MkDocs\nsite/\n',
        os.path.join(root_dir, '.env.example'): 'OLLAMA_MODEL=llama3\nFAISS_INDEX_PATH=./faiss_index\nDB_PATH=./metadata.db\n',
        os.path.join(root_dir, 'README.md'): '# RAG Assistant\n\nЛокальный интеллектуальный ассистент на основе RAG для поиска и анализа файлов (PDF, TXT, DOCX, PNG, JPG).\n\n## Установка\n```bash\npoetry install\n```\n\n## Запуск\n```bash\npython src/main.py\n```\n\n## Этапы разработки\n1. Базовый RAG (индексация, поиск).\n2. UI (PyQt5, чат).\n3. Интеграция с ОС (ПКМ).\n',
        os.path.join(root_dir, 'LICENSE'): 'MIT License\n\nCopyright (c) 2025 [Your Name]\n\nPermission is hereby granted, free of charge, to any person obtaining a copy...\n',  # Вставь полный текст MIT лицензии
        os.path.join(root_dir, 'pyproject.toml'): '''[tool.poetry]
name = "rag-assistant"
version = "0.1.0"
description = "Локальный ассистент на основе RAG для поиска и анализа файлов"
authors = ["Your Name <your.email@example.com>"]
license = "MIT"

[tool.poetry.dependencies]
python = "^3.10"
ollama = "*"
langchain = "*"
sentence-transformers = "*"
faiss-cpu = "*"
pyqt5 = "*"
pypdf2 = "*"
docx2txt = "*"
pillow = "*"
pytesseract = "*"
beautifulsoup4 = "*"
sqlalchemy = "*"
pywin32 = {version = "*", platform = "win32"}

[tool.poetry.dev-dependencies]
pytest = "*"
pytest-qt = "*"
mkdocs = "*"
mkdocs-material = "*"
pyinstaller = "*"

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"
''',
        os.path.join(root_dir, 'requirements.txt'): '# Зависимости (генерируются из pyproject.toml)\n# ollama\n# langchain\n# sentence-transformers\n# etc.\n',
        os.path.join(root_dir, '.github/workflows/ci.yml'): '''name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: "3.10"
      - name: Install dependencies
        run: pip install poetry && poetry install
      - name: Run tests
        run: poetry run pytest tests/ --cov=src/ --cov-report=xml
'''
    }

    for file_path, content in files_content.items():
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

    print(f"Структура проекта создана в текущей директории: {root_dir}")

if __name__ == "__main__":
    create_structure()