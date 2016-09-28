#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe train --solver=examples/siamese_ukbench/mnist_siamese_solver_finetune.prototxt$@
#--weights=models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel$@
