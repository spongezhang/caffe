#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe train --solver=examples/siamese_ukbench/mnist_siamese_solver_classification.prototxt $@
