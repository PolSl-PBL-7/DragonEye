#################
# CONFIGURATION #
#################

PYTHONPATH:=.:${PYTHONPATH}
DEV_CONTAINER:="dragoneye:dev-local"

###########
# TESTING #
###########

.PHONY: test
test:
	python -m pytest .

.PHONY: flake
flake:
	python -m flake8

.PHONY: mypy
mypy:
	python -m mypy .

.PHONY: autopep
autopep:
	find . -name "*.py" \
		| xargs python3 -m autopep8 --in-place

.PHONY: local-ci
local-ci:
	docker build \
		--file ./Dockerfile.CI \
		--rm \
		.

###########################
# DEPENDENCIES MANAGEMENT #
###########################

.PHONY: deps-ci
deps-ci:
	./scripts/bootstrap-ci.sh

.PHONY: deps
deps:
	./scripts/bootstrap-container.sh

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
