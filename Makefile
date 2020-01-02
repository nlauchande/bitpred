setup:
	docker build -t bitpred-docker -f Dockerfile .
run:
	mlflow run .
