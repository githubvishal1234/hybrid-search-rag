.PHONY: install data ingest index ask serve ui eval eval-dev eval-test eval-full judge golden-draft golden-review golden-add golden-check qdrant qdrant-down lint test

install:
	uv sync

data:
	uv run python scripts/download_data.py

ingest:
	uv run python -m hybrid_rag.ingestion.pipeline

index:
	uv run python -m hybrid_rag.indexing.build

ask:
	uv run python -m hybrid_rag.cli "$(Q)"

serve:
	uv run uvicorn hybrid_rag.api.main:app --port 8000

ui:
	uv run streamlit run ui/app.py

eval:
	uv run python eval/run_eval.py --sweep

eval-full:
	uv run python eval/run_eval.py --generate

eval-dev:
	uv run python eval/run_eval.py --split dev --sweep --label dev

eval-test:
	uv run python eval/run_eval.py --split test --label test

judge:
	uv run python eval/judge.py $(CSV)

golden-draft:
	uv run python scripts/build_golden_set.py

golden-review:
	uv run python scripts/review_golden_set.py

golden-add:
	uv run python scripts/review_golden_set.py --add

golden-check:
	uv run python scripts/review_golden_set.py --check

qdrant:
	docker compose up -d qdrant

qdrant-down:
	docker compose down

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest -q
