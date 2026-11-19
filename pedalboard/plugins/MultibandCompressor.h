/*
 * pedalboard
 * Copyright 2021 Spotify AB
 *
 * Licensed under the GNU Public License, Version 3.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    https://www.gnu.org/licenses/gpl-3.0.html
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "../JuceHeader.h"
#include "../Plugin.h"
#include "BrickwallLimiter.h"

#include <array>
#include <atomic>
#include <cmath>
#include <memory>

namespace Pedalboard {

/**
 * The independently-configurable compressor settings for one band of a
 * MultibandCompressor. Instances are owned by the parent plugin and exposed to
 * Python as live views; assigning a property updates the band immediately.
 */
class MultibandCompressorBand {
public:
  float getThresholdDb() const noexcept {
    return thresholdDb_.load(std::memory_order_relaxed);
  }
  void setThresholdDb(float value) {
    if (!std::isfinite(value))
      throw std::range_error("threshold_db must be a finite number");
    thresholdDb_.store(value, std::memory_order_relaxed);
  }

  float getRatio() const noexcept {
    return ratio_.load(std::memory_order_relaxed);
  }
  void setRatio(float value) {
    if (!std::isfinite(value) || value < 1.0f)
      throw std::range_error("ratio must be a finite value >= 1.0");
    ratio_.store(value, std::memory_order_relaxed);
  }

  float getAttackMs() const noexcept {
    return attackMs_.load(std::memory_order_relaxed);
  }
  void setAttackMs(float value) {
    if (!std::isfinite(value) || value < 0.0f)
      throw std::range_error("attack_ms must be a finite value >= 0");
    attackMs_.store(value, std::memory_order_relaxed);
  }

  float getReleaseMs() const noexcept {
    return releaseMs_.load(std::memory_order_relaxed);
  }
  void setReleaseMs(float value) {
    if (!std::isfinite(value) || value <= 0.0f)
      throw std::range_error("release_ms must be a finite value > 0");
    releaseMs_.store(value, std::memory_order_relaxed);
  }

private:
  template <typename SampleType> friend class MultibandCompressor;

  std::atomic<float> thresholdDb_{0.0f};
  std::atomic<float> ratio_{1.0f};
  std::atomic<float> attackMs_{1.0f};
  std::atomic<float> releaseMs_{100.0f};
};

/**
 * A three-band compressor with Linkwitz-Riley 4th-order (24 dB/octave)
 * crossovers and a master brick-wall limiter.
 *
 * Audio is split into low, mid, and high bands, each with independent
 * threshold/ratio/attack/release settings. The bands are then summed and passed
 * through a lookahead limiter that enforces the requested output ceiling.
 *
 * Processing is block-based and allocation-free after prepare(), so memory use
 * is bounded by the configured buffer size rather than the length of the input.
 * The default limiter does not use oversampling; set output_true_peak=True
 * when the extra CPU cost is acceptable.
 */
template <typename SampleType> class MultibandCompressor : public Plugin {
public:
  MultibandCompressor()
      : lowBand_(std::make_shared<MultibandCompressorBand>()),
        midBand_(std::make_shared<MultibandCompressorBand>()),
        highBand_(std::make_shared<MultibandCompressorBand>()) {
    limiter_.setCeilingDb(-1.0f);
    limiter_.setReleaseMs(100.0f);
    limiter_.setLookaheadMs(5.0f);
    limiter_.setTruePeak(false);
  }

  float getLowMidCrossoverHz() const noexcept {
    return lowMidCrossoverHz_.load(std::memory_order_relaxed);
  }
  void setLowMidCrossoverHz(float value) {
    if (!std::isfinite(value) || value <= 0.0f)
      throw std::range_error("low_mid_crossover_hz must be finite and > 0");
    lowMidCrossoverHz_.store(value, std::memory_order_relaxed);
  }

  float getMidHighCrossoverHz() const noexcept {
    return midHighCrossoverHz_.load(std::memory_order_relaxed);
  }
  void setMidHighCrossoverHz(float value) {
    if (!std::isfinite(value) || value <= 0.0f)
      throw std::range_error("mid_high_crossover_hz must be finite and > 0");
    midHighCrossoverHz_.store(value, std::memory_order_relaxed);
  }

  void validateCrossoverOrder() const {
    if (getLowMidCrossoverHz() >= getMidHighCrossoverHz())
      throw std::range_error(
          "low_mid_crossover_hz must be less than mid_high_crossover_hz");
  }

  float getOutputCeilingDb() const noexcept {
    return limiter_.getCeilingDb();
  }
  void setOutputCeilingDb(float value) { limiter_.setCeilingDb(value); }

  float getOutputReleaseMs() const noexcept {
    return limiter_.getReleaseMs();
  }
  void setOutputReleaseMs(float value) { limiter_.setReleaseMs(value); }

  float getOutputLookaheadMs() const noexcept {
    return limiter_.getLookaheadMs();
  }
  void setOutputLookaheadMs(float value) { limiter_.setLookaheadMs(value); }

  bool getOutputTruePeak() const noexcept { return limiter_.getTruePeak(); }
  void setOutputTruePeak(bool value) { limiter_.setTruePeak(value); }

  std::shared_ptr<MultibandCompressorBand> getLowBand() const {
    return lowBand_;
  }
  std::shared_ptr<MultibandCompressorBand> getMidBand() const {
    return midBand_;
  }
  std::shared_ptr<MultibandCompressorBand> getHighBand() const {
    return highBand_;
  }

  virtual void prepare(const juce::dsp::ProcessSpec &spec) override {
    validateCrossoverOrder();
    bool specChanged =
        (spec.sampleRate != preparedSpec_.sampleRate ||
         preparedSpec_.maximumBlockSize < spec.maximumBlockSize ||
         spec.numChannels != preparedSpec_.numChannels);
    float lowMid = lowMidCrossoverHz_.load(std::memory_order_relaxed);
    float midHigh = midHighCrossoverHz_.load(std::memory_order_relaxed);
    bool crossoverChanged =
        (lowMid != activeLowMidCrossoverHz_ ||
         midHigh != activeMidHighCrossoverHz_);

    if (specChanged) {
      firstCrossover_.prepare(spec);
      secondCrossover_.prepare(spec);
      bandCompressors_[0].prepare(spec);
      bandCompressors_[1].prepare(spec);
      bandCompressors_[2].prepare(spec);
      lowBandBuffer_.setSize(static_cast<int>(spec.numChannels),
                             static_cast<int>(spec.maximumBlockSize));
      midBandBuffer_.setSize(static_cast<int>(spec.numChannels),
                             static_cast<int>(spec.maximumBlockSize));
      highBandBuffer_.setSize(static_cast<int>(spec.numChannels),
                              static_cast<int>(spec.maximumBlockSize));
      preparedSpec_ = spec;
    }

    if (specChanged || crossoverChanged) {
      // Keep crossovers valid at low sample rates instead of passing values
      // above Nyquist into JUCE's assertion-only validation.
      const float maxCutoff =
          std::max(1.0f, static_cast<float>(spec.sampleRate) * 0.49f);
      float safeMidHigh = juce::jlimit(1.0f, maxCutoff, midHigh);
      float safeLowMid = juce::jlimit(
          1.0f, std::max(1.0f, safeMidHigh - 1.0f), lowMid);
      firstCrossover_.setCutoffFrequency(safeLowMid);
      secondCrossover_.setCutoffFrequency(safeMidHigh);
      activeLowMidCrossoverHz_ = lowMid;
      activeMidHighCrossoverHz_ = midHigh;
    }

    limiter_.prepare(spec);
  }

  virtual int
  process(const juce::dsp::ProcessContextReplacing<SampleType> &context) override {
    juce::ScopedNoDenormals noDenormals;
    auto ioBlock = context.getOutputBlock();
    const int numChannels = static_cast<int>(ioBlock.getNumChannels());
    const int numSamples = static_cast<int>(ioBlock.getNumSamples());

    jassert(numChannels <= lowBandBuffer_.getNumChannels());
    jassert(numSamples <= lowBandBuffer_.getNumSamples());

    for (int channel = 0; channel < numChannels; ++channel) {
      auto *input = ioBlock.getChannelPointer(static_cast<size_t>(channel));
      auto *low = lowBandBuffer_.getWritePointer(channel);
      auto *mid = midBandBuffer_.getWritePointer(channel);
      auto *high = highBandBuffer_.getWritePointer(channel);

      for (int i = 0; i < numSamples; ++i) {
        SampleType lowSample;
        SampleType highMidSample;
        firstCrossover_.processSample(channel, input[i], lowSample,
                                      highMidSample);

        SampleType midSample;
        SampleType highSample;
        secondCrossover_.processSample(channel, highMidSample, midSample,
                                       highSample);

        low[i] = lowSample;
        mid[i] = midSample;
        high[i] = highSample;
      }
    }

    bandCompressors_[0].process(
        lowBandBuffer_.getArrayOfWritePointers(), numChannels, numSamples,
        *lowBand_);
    bandCompressors_[1].process(
        midBandBuffer_.getArrayOfWritePointers(), numChannels, numSamples,
        *midBand_);
    bandCompressors_[2].process(
        highBandBuffer_.getArrayOfWritePointers(), numChannels, numSamples,
        *highBand_);

    for (int channel = 0; channel < numChannels; ++channel) {
      auto *output = ioBlock.getChannelPointer(static_cast<size_t>(channel));
      juce::FloatVectorOperations::copy(
          output, lowBandBuffer_.getReadPointer(channel), numSamples);
      juce::FloatVectorOperations::add(
          output, midBandBuffer_.getReadPointer(channel), numSamples);
      juce::FloatVectorOperations::add(
          output, highBandBuffer_.getReadPointer(channel), numSamples);
    }

    juce::dsp::ProcessContextReplacing<SampleType> limiterContext(ioBlock);
    return limiter_.process(limiterContext);
  }

  virtual void reset() override {
    firstCrossover_.reset();
    secondCrossover_.reset();
    for (auto &compressor : bandCompressors_)
      compressor.reset();
    limiter_.reset();
  }

  virtual int getLatencyHint() override {
    return limiter_.getLatencyHint();
  }

private:
  class BandCompressor {
  public:
    void prepare(const juce::dsp::ProcessSpec &spec) {
      juce::dsp::ProcessSpec monoSpec = spec;
      monoSpec.numChannels = 1;
      envelope_.prepare(monoSpec);
      prepared_ = true;
      reset();
    }

    void reset() { envelope_.reset(); }

    void process(SampleType *const *channels, int numChannels,
                 int numSamples, const MultibandCompressorBand &band) {
      if (!prepared_ || numSamples <= 0)
        return;

      float thresholdDb = band.getThresholdDb();
      float ratio = std::max(1.0f, band.getRatio());
      float attackMs = band.getAttackMs();
      float releaseMs = std::max(0.001f, band.getReleaseMs());

      if (attackMs != activeAttackMs_) {
        envelope_.setAttackTime(attackMs);
        activeAttackMs_ = attackMs;
      }
      if (releaseMs != activeReleaseMs_) {
        envelope_.setReleaseTime(releaseMs);
        activeReleaseMs_ = releaseMs;
      }

      const SampleType threshold =
          juce::Decibels::decibelsToGain(thresholdDb, SampleType(-200.0));
      const SampleType thresholdInverse =
          SampleType(1.0) / std::max(threshold, SampleType(1e-20));
      const SampleType ratioInverse = SampleType(1.0) / ratio;

      for (int i = 0; i < numSamples; ++i) {
        SampleType peak = SampleType(0);
        for (int channel = 0; channel < numChannels; ++channel)
          peak = std::max(peak, std::abs(channels[channel][i]));

        const SampleType envelope = envelope_.processSample(0, peak);
        SampleType gain = SampleType(1.0);
        if (envelope >= threshold) {
          gain = std::pow(envelope * thresholdInverse,
                          ratioInverse - SampleType(1.0));
        }

        for (int channel = 0; channel < numChannels; ++channel)
          channels[channel][i] *= gain;
      }
    }

  private:
    juce::dsp::BallisticsFilter<SampleType> envelope_;
    float activeAttackMs_ = -1.0f;
    float activeReleaseMs_ = -1.0f;
    bool prepared_ = false;
  };

  std::atomic<float> lowMidCrossoverHz_{250.0f};
  std::atomic<float> midHighCrossoverHz_{3000.0f};
  float activeLowMidCrossoverHz_ = 0.0f;
  float activeMidHighCrossoverHz_ = 0.0f;

  juce::dsp::LinkwitzRileyFilter<SampleType> firstCrossover_;
  juce::dsp::LinkwitzRileyFilter<SampleType> secondCrossover_;

  std::shared_ptr<MultibandCompressorBand> lowBand_;
  std::shared_ptr<MultibandCompressorBand> midBand_;
  std::shared_ptr<MultibandCompressorBand> highBand_;
  std::array<BandCompressor, 3> bandCompressors_;

  juce::AudioBuffer<SampleType> lowBandBuffer_;
  juce::AudioBuffer<SampleType> midBandBuffer_;
  juce::AudioBuffer<SampleType> highBandBuffer_;
  BrickwallLimiter<SampleType> limiter_;

  juce::dsp::ProcessSpec preparedSpec_{};
};

inline void init_multiband_compressor(py::module &m) {
  auto band = py::class_<MultibandCompressorBand,
                         std::shared_ptr<MultibandCompressorBand>>(
      m, "MultibandCompressorBand",
      "Live parameter view for one band of a "
      ":class:`MultibandCompressor`. Obtained from "
      "``MultibandCompressor.low_band``, ``.mid_band``, or ``.high_band``.");

  band.def_property("threshold_db",
                    &MultibandCompressorBand::getThresholdDb,
                    &MultibandCompressorBand::setThresholdDb)
      .def_property("ratio", &MultibandCompressorBand::getRatio,
                    &MultibandCompressorBand::setRatio)
      .def_property("attack_ms", &MultibandCompressorBand::getAttackMs,
                    &MultibandCompressorBand::setAttackMs)
      .def_property("release_ms", &MultibandCompressorBand::getReleaseMs,
                    &MultibandCompressorBand::setReleaseMs)
      .def("__repr__", [](const MultibandCompressorBand &band) {
        std::ostringstream ss;
        ss << "<pedalboard.MultibandCompressorBand"
           << " threshold_db=" << band.getThresholdDb()
           << " ratio=" << band.getRatio()
           << " attack_ms=" << band.getAttackMs()
           << " release_ms=" << band.getReleaseMs() << ">";
        return ss.str();
      });

  py::class_<MultibandCompressor<float>, Plugin,
             std::shared_ptr<MultibandCompressor<float>>>(
      m, "MultibandCompressor",
      R"(A three-band compressor with 4th-order Linkwitz-Riley crossovers and
a master brick-wall limiter.

The input is split into low, mid, and high bands at
``low_mid_crossover_hz`` and ``mid_high_crossover_hz``. Each band has
independent ``threshold_db``, ``ratio``, ``attack_ms``, and ``release_ms``
controls, available both directly on this object (for example,
``low_threshold_db``) and through the ``low_band``, ``mid_band``, and
``high_band`` parameter views.

After compression, the bands are summed and passed through a lookahead
``BrickwallLimiter``. The default limiter uses sample-peak detection; set
``output_true_peak=True`` to enable 4x oversampled true-peak detection at a
higher CPU cost.

Processing is block-based and allocation-free after preparation, with memory
use proportional to ``buffer_size`` rather than track length. This makes the
effect suitable for long podcast episodes and streaming workflows.

Example::

    from pedalboard import MultibandCompressor

    compressor = MultibandCompressor(
        low_mid_crossover_hz=200,
        mid_high_crossover_hz=3000,
        low_threshold_db=-24,
        low_ratio=3,
        mid_threshold_db=-20,
        mid_ratio=2,
        high_threshold_db=-18,
        high_ratio=2,
        output_ceiling_db=-1,
    )
    output = compressor(audio, sample_rate)
)")
      .def(py::init([](
               float lowMidCrossoverHz, float midHighCrossoverHz,
               float lowThresholdDb, float lowRatio, float lowAttackMs,
               float lowReleaseMs, float midThresholdDb, float midRatio,
               float midAttackMs, float midReleaseMs, float highThresholdDb,
               float highRatio, float highAttackMs, float highReleaseMs,
               float outputCeilingDb, float outputReleaseMs,
               float outputLookaheadMs, bool outputTruePeak) {
             auto plugin = std::make_unique<MultibandCompressor<float>>();
             plugin->setMidHighCrossoverHz(midHighCrossoverHz);
             plugin->setLowMidCrossoverHz(lowMidCrossoverHz);
             plugin->validateCrossoverOrder();

             plugin->getLowBand()->setThresholdDb(lowThresholdDb);
             plugin->getLowBand()->setRatio(lowRatio);
             plugin->getLowBand()->setAttackMs(lowAttackMs);
             plugin->getLowBand()->setReleaseMs(lowReleaseMs);
             plugin->getMidBand()->setThresholdDb(midThresholdDb);
             plugin->getMidBand()->setRatio(midRatio);
             plugin->getMidBand()->setAttackMs(midAttackMs);
             plugin->getMidBand()->setReleaseMs(midReleaseMs);
             plugin->getHighBand()->setThresholdDb(highThresholdDb);
             plugin->getHighBand()->setRatio(highRatio);
             plugin->getHighBand()->setAttackMs(highAttackMs);
             plugin->getHighBand()->setReleaseMs(highReleaseMs);

             plugin->setOutputCeilingDb(outputCeilingDb);
             plugin->setOutputReleaseMs(outputReleaseMs);
             plugin->setOutputLookaheadMs(outputLookaheadMs);
             plugin->setOutputTruePeak(outputTruePeak);
             return plugin;
           }),
           py::arg("low_mid_crossover_hz") = 250.0f,
           py::arg("mid_high_crossover_hz") = 3000.0f,
           py::arg("low_threshold_db") = 0.0f, py::arg("low_ratio") = 1.0f,
           py::arg("low_attack_ms") = 1.0f,
           py::arg("low_release_ms") = 100.0f,
           py::arg("mid_threshold_db") = 0.0f, py::arg("mid_ratio") = 1.0f,
           py::arg("mid_attack_ms") = 1.0f,
           py::arg("mid_release_ms") = 100.0f,
           py::arg("high_threshold_db") = 0.0f, py::arg("high_ratio") = 1.0f,
           py::arg("high_attack_ms") = 1.0f,
           py::arg("high_release_ms") = 100.0f,
           py::arg("output_ceiling_db") = -1.0f,
           py::arg("output_release_ms") = 100.0f,
           py::arg("output_lookahead_ms") = 5.0f,
           py::arg("output_true_peak") = false)
      .def("__repr__",
           [](const MultibandCompressor<float> &plugin) {
             std::ostringstream ss;
             ss << "<pedalboard.MultibandCompressor"
                << " low_mid_crossover_hz=" << plugin.getLowMidCrossoverHz()
                << " mid_high_crossover_hz=" << plugin.getMidHighCrossoverHz()
                << " low_threshold_db="
                << plugin.getLowBand()->getThresholdDb()
                << " mid_threshold_db="
                << plugin.getMidBand()->getThresholdDb()
                << " high_threshold_db="
                << plugin.getHighBand()->getThresholdDb()
                << " output_ceiling_db=" << plugin.getOutputCeilingDb()
                << " at " << &plugin << ">";
             return ss.str();
           })
      .def_property("low_mid_crossover_hz",
                    &MultibandCompressor<float>::getLowMidCrossoverHz,
                    &MultibandCompressor<float>::setLowMidCrossoverHz)
      .def_property("mid_high_crossover_hz",
                    &MultibandCompressor<float>::getMidHighCrossoverHz,
                    &MultibandCompressor<float>::setMidHighCrossoverHz)
      .def_property("low_crossover_hz",
                    &MultibandCompressor<float>::getLowMidCrossoverHz,
                    &MultibandCompressor<float>::setLowMidCrossoverHz)
      .def_property("high_crossover_hz",
                    &MultibandCompressor<float>::getMidHighCrossoverHz,
                    &MultibandCompressor<float>::setMidHighCrossoverHz)
      .def_property("low_band", &MultibandCompressor<float>::getLowBand,
                    nullptr, py::return_value_policy::reference_internal)
      .def_property("mid_band", &MultibandCompressor<float>::getMidBand,
                    nullptr, py::return_value_policy::reference_internal)
      .def_property("high_band", &MultibandCompressor<float>::getHighBand,
                    nullptr, py::return_value_policy::reference_internal)
      .def_property("low_threshold_db",
                    [](const MultibandCompressor<float> &plugin) {
                      return plugin.getLowBand()->getThresholdDb();
                    },
                    [](MultibandCompressor<float> &plugin, float value) {
                      plugin.getLowBand()->setThresholdDb(value);
                    })
      .def_property("low_ratio",
                    [](const MultibandCompressor<float> &plugin) {
                      return plugin.getLowBand()->getRatio();
                    },
                    [](MultibandCompressor<float> &plugin, float value) {
                      plugin.getLowBand()->setRatio(value);
                    })
      .def_property("low_attack_ms",
                    [](const MultibandCompressor<float> &plugin) {
                      return plugin.getLowBand()->getAttackMs();
                    },
                    [](MultibandCompressor<float> &plugin, float value) {
                      plugin.getLowBand()->setAttackMs(value);
                    })
      .def_property("low_release_ms",
                    [](const MultibandCompressor<float> &plugin) {
                      return plugin.getLowBand()->getReleaseMs();
                    },
                    [](MultibandCompressor<float> &plugin, float value) {
                      plugin.getLowBand()->setReleaseMs(value);
                    })
      .def_property("mid_threshold_db",
                    [](const MultibandCompressor<float> &plugin) {
                      return plugin.getMidBand()->getThresholdDb();
                    },
                    [](MultibandCompressor<float> &plugin, float value) {
                      plugin.getMidBand()->setThresholdDb(value);
                    })
      .def_property("mid_ratio",
                    [](const MultibandCompressor<float> &plugin) {
                      return plugin.getMidBand()->getRatio();
                    },
                    [](MultibandCompressor<float> &plugin, float value) {
                      plugin.getMidBand()->setRatio(value);
                    })
      .def_property("mid_attack_ms",
                    [](const MultibandCompressor<float> &plugin) {
                      return plugin.getMidBand()->getAttackMs();
                    },
                    [](MultibandCompressor<float> &plugin, float value) {
                      plugin.getMidBand()->setAttackMs(value);
                    })
      .def_property("mid_release_ms",
                    [](const MultibandCompressor<float> &plugin) {
                      return plugin.getMidBand()->getReleaseMs();
                    },
                    [](MultibandCompressor<float> &plugin, float value) {
                      plugin.getMidBand()->setReleaseMs(value);
                    })
      .def_property("high_threshold_db",
                    [](const MultibandCompressor<float> &plugin) {
                      return plugin.getHighBand()->getThresholdDb();
                    },
                    [](MultibandCompressor<float> &plugin, float value) {
                      plugin.getHighBand()->setThresholdDb(value);
                    })
      .def_property("high_ratio",
                    [](const MultibandCompressor<float> &plugin) {
                      return plugin.getHighBand()->getRatio();
                    },
                    [](MultibandCompressor<float> &plugin, float value) {
                      plugin.getHighBand()->setRatio(value);
                    })
      .def_property("high_attack_ms",
                    [](const MultibandCompressor<float> &plugin) {
                      return plugin.getHighBand()->getAttackMs();
                    },
                    [](MultibandCompressor<float> &plugin, float value) {
                      plugin.getHighBand()->setAttackMs(value);
                    })
      .def_property("high_release_ms",
                    [](const MultibandCompressor<float> &plugin) {
                      return plugin.getHighBand()->getReleaseMs();
                    },
                    [](MultibandCompressor<float> &plugin, float value) {
                      plugin.getHighBand()->setReleaseMs(value);
                    })
      .def_property("output_ceiling_db",
                    &MultibandCompressor<float>::getOutputCeilingDb,
                    &MultibandCompressor<float>::setOutputCeilingDb)
      .def_property("output_release_ms",
                    &MultibandCompressor<float>::getOutputReleaseMs,
                    &MultibandCompressor<float>::setOutputReleaseMs)
      .def_property("output_lookahead_ms",
                    &MultibandCompressor<float>::getOutputLookaheadMs,
                    &MultibandCompressor<float>::setOutputLookaheadMs)
      .def_property("output_true_peak",
                    &MultibandCompressor<float>::getOutputTruePeak,
                    &MultibandCompressor<float>::setOutputTruePeak);

  m.attr("ThreeBandCompressor") = m.attr("MultibandCompressor");
}

} // namespace Pedalboard
