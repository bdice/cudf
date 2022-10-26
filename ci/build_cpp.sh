#!/bin/bash

set -euo pipefail

source rapids-env-update

rapids-print-env

rapids-logger "Begin cpp build"

rapids-mamba-retry mambabuild conda/recipes/libcudf

rapids-upload-conda-to-s3 cpp
