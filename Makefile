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

.PHONY: flake
flake:
	python3 -m flake8

# .PHONY: mypy
# mypy:
# 	python3 -m mypy .

.PHONY: autopep
autopep:
	autopep8 --in-place --recursive .

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

.PHONY: install-container-toolkit
install-container-toolkit:
	./scripts/install-container-toolkit.sh

.PHONY: deps
deps:
	./scripts/bootstrap.sh

.PHONY: freeze
freeze:
	python3 -m pip list --format=freeze \
		| grep -v "^tensorflow" \
		> ./requirements.txt
	git add ./requirements.txt
