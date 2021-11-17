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
	pip list --format=freeze \
		| grep -v "^tensorflow" \
		> ./requirements.txt
	git add ./requirements.txt

.PHONY: freeze-conda
freeze-conda:
	conda env export > environment.yml
	python3 ./scripts/parse-yml-to-txt.py \
		| grep -v "^tensorflow" \
		> requirements.txt
	rm environment.yml

.PHONY: local-ci
local-ci:
	docker build --rm .
