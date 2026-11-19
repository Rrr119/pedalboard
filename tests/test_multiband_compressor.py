import numpy as np
import pytest

from pedalboard import MultibandCompressor, ThreeBandCompressor

from .utils import db_to_gain


def _sine(
    sample_rate: int,
    frequency_hz: float,
    amplitude: float = 0.1,
    duration_seconds: float = 2.0,
    num_channels: int = 1,
) -> np.ndarray:
    samples = np.arange(int(sample_rate * duration_seconds), dtype=np.float32)
    wave = amplitude * np.sin(2.0 * np.pi * frequency_hz * samples / sample_rate).astype(np.float32)
    if num_channels > 1:
        return np.stack([wave] * num_channels)
    return wave


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def test_parameters_are_independent():
    compressor = MultibandCompressor(
        low_mid_crossover_hz=180,
        mid_high_crossover_hz=4000,
        low_threshold_db=-24,
        low_ratio=3,
        low_attack_ms=20,
        low_release_ms=200,
        mid_threshold_db=-20,
        mid_ratio=2.5,
        mid_attack_ms=10,
        mid_release_ms=120,
        high_threshold_db=-18,
        high_ratio=2,
        high_attack_ms=5,
        high_release_ms=80,
        output_ceiling_db=-1.5,
        output_release_ms=75,
        output_lookahead_ms=4,
        output_true_peak=False,
    )

    assert compressor.low_mid_crossover_hz == pytest.approx(180)
    assert compressor.mid_high_crossover_hz == pytest.approx(4000)
    assert compressor.low_crossover_hz == pytest.approx(180)
    assert compressor.high_crossover_hz == pytest.approx(4000)

    assert compressor.low_threshold_db == pytest.approx(-24)
    assert compressor.low_ratio == pytest.approx(3)
    assert compressor.low_attack_ms == pytest.approx(20)
    assert compressor.low_release_ms == pytest.approx(200)
    assert compressor.mid_threshold_db == pytest.approx(-20)
    assert compressor.mid_ratio == pytest.approx(2.5)
    assert compressor.mid_attack_ms == pytest.approx(10)
    assert compressor.mid_release_ms == pytest.approx(120)
    assert compressor.high_threshold_db == pytest.approx(-18)
    assert compressor.high_ratio == pytest.approx(2)
    assert compressor.high_attack_ms == pytest.approx(5)
    assert compressor.high_release_ms == pytest.approx(80)

    compressor.low_band.threshold_db = -30
    compressor.mid_band.ratio = 4
    compressor.high_band.release_ms = 50

    assert compressor.low_threshold_db == pytest.approx(-30)
    assert compressor.mid_ratio == pytest.approx(4)
    assert compressor.high_release_ms == pytest.approx(50)
    assert compressor.output_ceiling_db == pytest.approx(-1.5)
    assert compressor.output_release_ms == pytest.approx(75)
    assert compressor.output_lookahead_ms == pytest.approx(4)
    assert compressor.output_true_peak is False

    assert "low_mid_crossover_hz=180" in repr(compressor)
    assert "threshold_db=-30" in repr(compressor.low_band)


@pytest.mark.parametrize("sample_rate", [22050, 44100, 48000])
@pytest.mark.parametrize("num_channels", [1, 2])
def test_crossover_reconstruction_is_flat(sample_rate, num_channels):
    samples = np.arange(sample_rate * 2, dtype=np.float32)
    signal = (
        0.08 * np.sin(2 * np.pi * 100 * samples / sample_rate)
        + 0.08 * np.sin(2 * np.pi * 1000 * samples / sample_rate)
        + 0.08 * np.sin(2 * np.pi * 7000 * samples / sample_rate)
    ).astype(np.float32)
    if num_channels == 2:
        signal = np.stack([signal, signal * 0.8])

    compressor = MultibandCompressor(output_ceiling_db=0.0)
    output = compressor(signal, sample_rate)

    assert output.shape == signal.shape
    assert np.all(np.isfinite(output))
    np.testing.assert_allclose(_rms(output), _rms(signal), rtol=0.03)


@pytest.mark.parametrize(
    "compressed_frequency_hz,uncompressed_frequency_hz",
    [(100, 7000), (1000, 7000), (7000, 100)],
)
def test_band_compression_is_independent(compressed_frequency_hz, uncompressed_frequency_hz):
    sample_rate = 48000
    band = (
        "low"
        if compressed_frequency_hz < 250
        else "mid"
        if compressed_frequency_hz < 3000
        else "high"
    )
    signal = _sine(sample_rate, compressed_frequency_hz, amplitude=0.5)
    other_signal = _sine(sample_rate, uncompressed_frequency_hz, amplitude=0.5)

    transparent = MultibandCompressor(output_ceiling_db=0.0)
    compressed = MultibandCompressor(output_ceiling_db=0.0)
    setattr(compressed, f"{band}_threshold_db", -30.0)
    setattr(compressed, f"{band}_ratio", 10.0)
    setattr(compressed, f"{band}_attack_ms", 1.0)
    setattr(compressed, f"{band}_release_ms", 50.0)

    transparent_output = transparent(signal, sample_rate)
    compressed_output = compressed(signal, sample_rate)
    assert _rms(compressed_output) < _rms(transparent_output) * 0.9

    transparent_other = transparent(other_signal, sample_rate)
    compressed_other = compressed(other_signal, sample_rate)
    np.testing.assert_allclose(_rms(compressed_other), _rms(transparent_other), rtol=0.05)


@pytest.mark.parametrize("num_channels", [1, 2])
def test_output_limiter_enforces_ceiling(num_channels):
    sample_rate = 48000
    signal = _sine(sample_rate, 440, amplitude=1.0, num_channels=num_channels)
    signal = signal * np.float32(db_to_gain(12.0))
    ceiling_db = -1.0

    output = MultibandCompressor(output_ceiling_db=ceiling_db)(signal, sample_rate)

    assert np.all(np.isfinite(output))
    assert np.max(np.abs(output)) <= db_to_gain(ceiling_db) + 1e-3


@pytest.mark.parametrize("sample_rate", [22050, 44100, 48000, 96000])
@pytest.mark.parametrize("num_channels", [1, 2])
def test_processes_supported_layouts(sample_rate, num_channels):
    signal = _sine(
        sample_rate,
        440,
        amplitude=0.25,
        duration_seconds=0.5,
        num_channels=num_channels,
    )
    output = MultibandCompressor(
        low_threshold_db=-24,
        low_ratio=3,
        mid_threshold_db=-20,
        mid_ratio=2,
        high_threshold_db=-18,
        high_ratio=2,
    )(signal, sample_rate)

    assert output.shape == signal.shape
    assert np.all(np.isfinite(output))
    assert np.max(np.abs(output)) <= 1.0


def test_parameter_validation():
    compressor = MultibandCompressor()

    with pytest.raises(ValueError):
        MultibandCompressor(low_mid_crossover_hz=4000, mid_high_crossover_hz=3000)
    with pytest.raises(ValueError):
        compressor.low_ratio = 0.5
    with pytest.raises(ValueError):
        compressor.mid_threshold_db = float("inf")
    with pytest.raises(ValueError):
        compressor.high_attack_ms = -1
    with pytest.raises(ValueError):
        compressor.high_band.release_ms = 0
    compressor.low_mid_crossover_hz = 4000
    with pytest.raises(ValueError):
        compressor(np.zeros(1000, dtype=np.float32), 48000)
    compressor.low_mid_crossover_hz = 250
    compressor.mid_high_crossover_hz = 100
    with pytest.raises(ValueError):
        compressor(np.zeros(1000, dtype=np.float32), 48000)
    with pytest.raises(ValueError):
        compressor.output_ceiling_db = float("nan")
    with pytest.raises(ValueError):
        compressor.output_release_ms = 0
    with pytest.raises(ValueError):
        compressor.output_lookahead_ms = 0

    reordered = MultibandCompressor()
    reordered.mid_high_crossover_hz = 200
    reordered.low_mid_crossover_hz = 100
    reordered(np.zeros(1000, dtype=np.float32), 48000)


def test_three_band_alias():
    assert ThreeBandCompressor is MultibandCompressor
