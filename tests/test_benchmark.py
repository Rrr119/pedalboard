#! /usr/bin/env python
#
# Copyright 2021 Spotify AB
#
# Licensed under the GNU Public License, Version 3.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.gnu.org/licenses/gpl-3.0.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import time

import numpy as np
import pytest

import pedalboard
from pedalboard import MultibandCompressor


class timer(object):
    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.end_time = time.time()

    def __float__(self):
        return float(self.end_time - self.start_time)

    def __str__(self):
        return str(float(self))

    def __repr__(self):
        return str(float(self))


@pytest.mark.skip
def test_pysox_performance_difference():
    try:
        import sox  # type: ignore
    except ImportError:
        pytest.skip("SoX is not installed")
    transformer = sox.Transformer()
    transformer.reverb()

    sr = 48000
    noise = np.random.rand(2, sr).astype(np.float32)

    pysox_measurements = []
    for _ in range(0, 10):
        with timer() as pysox_time_taken:
            transformer.build_array(input_array=noise, sample_rate_in=sr)
        pysox_measurements.append(float(pysox_time_taken))

    pedalboard_measurements = []
    for _ in range(0, 10):
        with timer() as pedalboard_time_taken:
            pedalboard.Reverb()(noise, sample_rate=sr)
        pedalboard_measurements.append(float(pedalboard_time_taken))

    average_pysox_time = np.mean(pysox_measurements)
    average_pedalboard_time = np.mean(pedalboard_measurements)

    # In local tests, Pedalboard is about 300x faster than PySoX.
    # This test ensures we're at least 100x faster to account for
    # variations across test run environments.
    assert average_pysox_time / average_pedalboard_time > 100


def test_multiband_compressor_30_minute_throughput():
    """Benchmark a realistic 30-minute podcast render in bounded memory."""
    sample_rate = 48000
    duration_seconds = 30 * 60
    chunk_seconds = 5
    samples_per_chunk = sample_rate * chunk_seconds

    samples = np.arange(samples_per_chunk, dtype=np.float32)
    chunk = np.stack(
        [
            0.1 * np.sin(2 * np.pi * 220 * samples / sample_rate),
            0.1 * np.sin(2 * np.pi * 440 * samples / sample_rate),
        ]
    ).astype(np.float32)
    compressor = MultibandCompressor(
        low_mid_crossover_hz=200,
        mid_high_crossover_hz=3000,
        low_threshold_db=-24,
        low_ratio=3,
        low_attack_ms=20,
        low_release_ms=200,
        mid_threshold_db=-20,
        mid_ratio=2,
        mid_attack_ms=10,
        mid_release_ms=120,
        high_threshold_db=-18,
        high_ratio=2,
        high_attack_ms=5,
        high_release_ms=80,
        output_ceiling_db=-1,
    )

    num_chunks = duration_seconds // chunk_seconds
    start = time.perf_counter()
    for chunk_index in range(num_chunks):
        compressor(chunk, sample_rate, reset=(chunk_index == 0))
    elapsed = time.perf_counter() - start

    realtime_factor = duration_seconds / elapsed
    print(f"\n30 min stereo render: {elapsed:.2f}s ({realtime_factor:.1f}x realtime)")

    # Generous enough for slow CI machines, but catches accidental
    # full-track buffering or per-sample filter setup regressions.
    assert elapsed < duration_seconds / 10
