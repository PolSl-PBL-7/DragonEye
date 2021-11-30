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
	python3 -m pytest .

# .PHONY: flake
# flake:
# 	python3 -m flake8

# .PHONY: mypy
# mypy:
# 	python3 -m mypy .

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

.PHONY: install-cuda
install-cuda:
	./scripts/install-cuda.sh

.PHONY: deps
deps:
	./scripts/bootstrap.sh

.PHONY: freeze
freeze:
	python3 -m pip list --format=freeze \
		| grep -v "^tensorflow" \
		> ./requirements.txt
	git add ./requirements.txt
