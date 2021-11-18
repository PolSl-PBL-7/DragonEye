PYTHONPATH:=.:${PYTHONPATH}

.PHONY: prerequisite-ci
prerequisite-ci:
	./scripts/bootstrap-ci.sh

.PHONY: test
test:
	python -m pytest .

.PHONY: flake
flake:
	python -m flake8

.PHONY: mypy
mypy:
	python -m mypy .

.PHONY: freeze
freeze:
	pip freeze | grep -v "^tensorflow" > ./requirements.txt
	git add ./requirements.txt

.PHONY: local-ci
local-ci:
	docker build --rm .
