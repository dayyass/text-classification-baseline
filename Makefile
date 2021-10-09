all:
	python -m text_clf --path_to_config config.yaml
load_data:
	python data/load_20newsgroups.py
coverage:
	coverage run -m unittest discover && coverage report -m
docker_build:
	docker image build -t text-classification-baseline .
docker_run:
	docker container run -it text-classification-baseline
pypi_packages:
	pip install --upgrade build twine
pypi_build:
	python -m build
pypi_twine:
	python -m twine upload --repository testpypi dist/*
clean:
	rm -rf models/model*
