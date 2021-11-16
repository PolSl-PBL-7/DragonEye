PYTHONPATH:=.:${PYTHONPATH}

.PHONY: prerequisite-ci
prerequisite-ci:
	apt-get update -y
	cat debian.depends | xargs apt-get install -y
	pip install pipenv
	sh -c ./scripts/replace.sh
	pipenv install --system --deploy --ignore-pipfile --dev

.PHONY: test
test:
	pytest .

.PHONY: flake
flake:
	flake8

.PHONY: commitlint
commitlint:
	npx

.PHONY: mypy
mypy:
	mypy .
