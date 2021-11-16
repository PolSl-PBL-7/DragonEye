FROM python:3.9

RUN mkdir /work
COPY . /work
WORKDIR /work

ENV PIP_EXTRA_INDEX_URL=https://snapshots.linaro.org/ldcg/python-cache/

RUN make prerequisite-ci

RUN make test
