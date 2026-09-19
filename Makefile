run_sanity:
	python -m flows.haw_multi_model_flow --config sanity.yaml

run_multi_model:
	python -m flows.haw_multi_model_flow

run_single_model:
	python -m flows.haw_multi_model_flow --config single_model_experiment.yaml
