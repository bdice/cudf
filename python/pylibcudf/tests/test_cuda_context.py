# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pytest
from cuda.bindings import driver

from rmm.pylibrmm.stream import DEFAULT_STREAM, PER_THREAD_DEFAULT_STREAM

import pylibcudf as plc


@pytest.mark.parametrize(
    "stream", [None, DEFAULT_STREAM, PER_THREAD_DEFAULT_STREAM]
)
def test_get_stream_initializes_thread_context(stream):
    assert driver.cuInit(0) == (driver.CUresult.CUDA_SUCCESS,)

    def get_stream():
        status, context = driver.cuCtxGetCurrent()
        assert status == driver.CUresult.CUDA_SUCCESS
        assert int(context) == 0
        plc.utils._get_stream(stream)
        status, context = driver.cuCtxGetCurrent()
        assert status == driver.CUresult.CUDA_SUCCESS
        assert int(context) != 0
        plc.utils._get_stream(stream)
        assert driver.cuCtxGetCurrent() == (status, context)
        assert driver.cuCtxPopCurrent() == (status, context)
        plc.utils._get_stream(stream)
        assert driver.cuCtxGetCurrent() == (status, context)

    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(get_stream).result()


def test_get_stream_preserves_current_context():
    assert driver.cuInit(0) == (driver.CUresult.CUDA_SUCCESS,)
    status, device = driver.cuDeviceGet(0)
    assert status == driver.CUresult.CUDA_SUCCESS
    status, context = driver.cuDevicePrimaryCtxRetain(device)
    assert status == driver.CUresult.CUDA_SUCCESS

    def get_stream():
        assert driver.cuCtxPushCurrent(context) == (
            driver.CUresult.CUDA_SUCCESS,
        )
        try:
            plc.utils._get_stream()
            assert driver.cuCtxGetCurrent() == (
                driver.CUresult.CUDA_SUCCESS,
                context,
            )
        finally:
            assert driver.cuCtxPopCurrent() == (
                driver.CUresult.CUDA_SUCCESS,
                context,
            )
        status, current = driver.cuCtxGetCurrent()
        assert status == driver.CUresult.CUDA_SUCCESS
        assert int(current) == 0

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(get_stream).result()
    finally:
        assert driver.cuDevicePrimaryCtxRelease(device) == (
            driver.CUresult.CUDA_SUCCESS,
        )


def test_empty_like_on_fresh_thread():
    column = plc.Column.from_arrow(pa.array([1, 2, 3]))
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(plc.copying.empty_like, column).result()
    assert result.size() == 0


def test_get_stream_initializes_cuda():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
from cuda.bindings import driver
import pylibcudf as plc

status, _ = driver.cuCtxGetCurrent()
assert status == driver.CUresult.CUDA_ERROR_NOT_INITIALIZED
plc.utils._get_stream()
status, context = driver.cuCtxGetCurrent()
assert status == driver.CUresult.CUDA_SUCCESS
assert int(context) != 0
""",
        ],
        check=True,
    )
