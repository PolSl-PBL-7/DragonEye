FROM python:3.9

RUN mkdir /work
COPY . /work
WORKDIR /work

RUN make prerequisite-ci

RUN make flake

RUN make mypy

RUN make test

RUN pip freeze
