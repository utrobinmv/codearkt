.PHONY: install black validate test publish

install:
	uv pip install -e .

black:
	uv run black container --line-length 100
	uv run black codearkt --line-length 100
	uv run black tests --line-length 100

validate:
	uv run black container --line-length 100
	uv run black codearkt --line-length 100
	uv run black tests --line-length 100
	uv run flake8 container
	uv run flake8 codearkt
	uv run flake8 tests
	uv run mypy container --strict --explicit-package-bases
	uv run mypy codearkt --strict --explicit-package-bases
	uv run mypy tests --strict --explicit-package-bases

test:
	uv run pytest -s

publish:
	uv build && uv publish