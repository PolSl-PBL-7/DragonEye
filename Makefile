PYTHONPATH:=.:${PYTHONPATH}

.PHONY: prerequisite-ci
prerequisite-ci:
	apt-get update -y
	cat debian.depends | xargs apt-get install -y
	sh -c ./scripts/install-non-depends.sh
	pip install pipenv
	sh -c ./scripts/replace.sh
	pipenv install --system --deploy --ignore-pipfile --dev

.PHONY: test
test:
	python -m pytest .

.PHONY: flake
flake:
	python -m flake8

.PHONY: mypy
mypy:
	python -m mypy .
