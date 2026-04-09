.PHONY: install run bootstrap serve monitor docker-up docker-down clean test

install:
	pip install -r requirements.txt

run:
	python run_pipeline.py --max-iterations 5

bootstrap:
	python run_pipeline.py --bootstrap-only

serve:
	python run_pipeline.py --serve

monitor:
	python run_pipeline.py --max-iterations 10 --monitor-interval 15

docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down -v

clean:
	rm -rf data/ artifacts/ mlflow.db
	rm -rf config/feature_repo/data/ config/feature_repo/registry.db
	rm -rf config/feature_repo/online_store.db config/feature_repo/__pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

test:
	python -m pytest tests/ -v
