setup:
	docker build -t bitpred-docker -f Dockerfile .

run:
	mlflow run .

serve:
 	mlflow models serve -m runs:/my-run-id/model-path

workbench-up:
	jupyter notebook

