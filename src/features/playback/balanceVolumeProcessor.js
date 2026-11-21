/* global AudioWorkletProcessor, registerProcessor, sampleRate */

/**
 * Real-time loudness leveler and peak limiter.
 *
 * The measurement follows BS.1770 K-weighting with the De Man coefficients for
 * the current sample rate. Control uses a 400 ms momentary window and a 3 s
 * short-term window, updated every 100 ms. Gain changes are intentionally slow
 * except for the first valid measurement, where the track starts at the
 * estimated level instead of fading in from unity.
 *
 * A five millisecond look-ahead limiter protects the output. The processor also
 * accepts a fixed per-track gain (the NewTube model): when fixedGainDb is set,
 * the measured auto gain is bypassed and the constant gain is limited only.
 */

class KFilter {
  constructor(highpass) {
    const k = Math.tan(Math.PI * (highpass ? 38.13547087602444 : 1681.974450955533) / sampleRate);
    const q = highpass ? 0.5003270373238773 : 0.7071752369554196;
    const a0 = 1 + k / q + k * k;
    const vh = Math.pow(10, 3.999843853973347 / 20);
    const vb = Math.pow(vh, 0.4996667741545416);
    this.b0 = highpass ? 1 : (vh + vb * k / q + k * k) / a0;
    this.b1 = highpass ? -2 : 2 * (k * k - vh) / a0;
    this.b2 = highpass ? 1 : (vh - vb * k / q + k * k) / a0;
    this.a1 = 2 * (k * k - 1) / a0;
    this.a2 = (1 - k / q + k * k) / a0;
    this.reset();
  }

  reset() {
    this.z1 = 0;
    this.z2 = 0;
  }

  tick(x) {
    const y = x * this.b0 + this.z1;
    this.z1 = x * this.b1 - y * this.a1 + this.z2;
    this.z2 = x * this.b2 - y * this.a2;
    return y;
  }
}

class BalanceVolumeProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    // Two channels, each with the K-weighting pre-filter and RLB high-pass.
    this.filters = [];
    for (let c = 0; c < 2; c += 1) {
      this.filters.push([new KFilter(false), new KFilter(true)]);
    }

    this.blockPowers = new Float64Array(30);
    this.delayFrames = Math.ceil(sampleRate * 0.005);
    this.capacity = this.delayFrames + 2;
    this.delay = [new Float32Array(this.capacity), new Float32Array(this.capacity)];
    this.peakValues = new Float64Array(this.capacity);
    this.peakFrames = new Float64Array(this.capacity);

    this.target = -18;
    this.strength = 1;
    this.volume = 1;
    this.maxBoostDb = 28;
    this.maxCutDb = -18;
    this.fixedGainDb = null;
    this.active = true;
    this.disposed = false;

    this.reset();
    this.port.onmessage = (event) => this.onMessage(event.data);
  }

  onMessage(data) {
    if (!data || typeof data !== 'object') return;

    if (data.type === 'dispose') {
      this.disposed = true;
      this.port.close();
      return;
    }

    if (data.type === 'reset') {
      this.reset();
      return;
    }

    if (data.type !== 'configure') return;

    if (Number.isFinite(data.target)) {
      this.target = Math.max(-24, Math.min(-14, data.target));
    }
    if (Number.isFinite(data.strength)) {
      this.strength = Math.max(0.4, Math.min(1, data.strength));
    }
    if (Number.isFinite(data.volume)) {
      this.volume = Math.max(0, Math.min(1, data.volume));
    }
    if (Number.isFinite(data.maxBoostDb)) {
      this.maxBoostDb = Math.max(0, Math.min(30, data.maxBoostDb));
    }
    if (Number.isFinite(data.maxCutDb)) {
      this.maxCutDb = Math.max(-30, Math.min(0, data.maxCutDb));
    }

    const nextFixed = Number.isFinite(data.fixedGainDb)
      ? Math.max(this.maxCutDb, Math.min(this.maxBoostDb, data.fixedGainDb))
      : null;
    if (nextFixed !== this.fixedGainDb) {
      this.fixedGainDb = nextFixed;
      if (nextFixed === null) {
        // Re-enter auto mode with a fresh estimate on the next valid block.
        this.hasMeasurement = false;
        this.desiredDb = 0;
      } else {
        this.gainDb = nextFixed;
        this.desiredDb = nextFixed;
        this.hasMeasurement = true;
      }
    }

    this.active = data.active === true;
  }

  reset() {
    this.blockPowers.fill(0);
    for (let c = 0; c < this.delay.length; c += 1) {
      this.delay[c].fill(0);
    }
    for (let c = 0; c < this.filters.length; c += 1) {
      this.filters[c][0].reset();
      this.filters[c][1].reset();
    }
    this.peakValues.fill(0);
    this.peakFrames.fill(0);

    this.blockIndex = 0;
    this.blockCount = 0;
    this.power = 0;
    this.frames = 0;
    this.frame = 0;
    this.head = 0;
    this.tail = 0;
    this.gainDb = this.fixedGainDb === null ? 0 : this.fixedGainDb;
    this.desiredDb = this.gainDb;
    this.limiterGain = 1;
    this.loudness = -100;
    this.hasMeasurement = this.fixedGainDb !== null;
  }

  analyze() {
    const meanPower = this.power / Math.max(this.frames, 1);
    this.power = 0;
    this.frames = 0;
    this.blockPowers[this.blockIndex] = meanPower;
    this.blockIndex = (this.blockIndex + 1) % this.blockPowers.length;
    this.blockCount = Math.min(this.blockPowers.length, this.blockCount + 1);
    if (this.blockCount < 4) return;

    let momentary = 0;
    let shortTerm = 0;
    for (let i = 0; i < this.blockCount; i += 1) {
      const index = (this.blockIndex - 1 - i + this.blockPowers.length) % this.blockPowers.length;
      const power = this.blockPowers[index];
      if (i < 4) momentary += power / 4;
      shortTerm += power / this.blockCount;
    }

    const momentaryLufs = -0.691 + 10 * Math.log10(Math.max(momentary, 1e-12));
    const shortTermLufs = -0.691 + 10 * Math.log10(Math.max(shortTerm, 1e-12));
    this.loudness = momentaryLufs;

    // Freeze in silence or near-silence instead of amplifying the noise floor.
    if (this.volume < 0.001 || momentaryLufs < -50) {
      this.desiredDb = this.gainDb;
      this.postMetrics();
      return;
    }

    if (this.fixedGainDb === null) {
      const reference = 0.75 * momentaryLufs + 0.25 * shortTermLufs;
      const desired = Math.max(
        this.maxCutDb,
        Math.min(this.maxBoostDb, (this.target - reference) * this.strength),
      );
      if (!this.hasMeasurement) {
        // Start the track at the estimated level. Subsequent corrections are slow.
        this.gainDb = desired;
        this.desiredDb = desired;
        this.hasMeasurement = true;
      } else if (Math.abs(desired - this.gainDb) > 1) {
        this.desiredDb = desired;
      }
    }

    this.postMetrics();
  }

  postMetrics() {
    this.port.postMessage({
      type: 'metrics',
      inputDb: this.loudness,
      gainDb: this.gainDb,
      desiredDb: this.desiredDb,
      reductionDb: 20 * Math.log10(Math.max(this.limiterGain, 1e-9)),
    });
  }

  process(inputs, outputs) {
    if (this.disposed) return false;

    const input = inputs[0];
    const output = outputs[0];
    if (!input || !output || !input.length || !output.length) return true;

    const frames = output[0].length;

    // The dry/wet crossfade on the main thread handles bypass. When inactive,
    // pass the signal through so a pause/resume transition never drops audio.
    if (!this.active) {
      for (let c = 0; c < output.length; c += 1) {
        const source = input.length > c ? input[c] : input[0];
        for (let i = 0; i < frames; i += 1) output[c][i] = source[i];
      }
      return true;
    }

    const release = 1 - Math.exp(-1 / (sampleRate * 0.08));
    const upStep = 1.5 / sampleRate;
    const downStep = -8 / sampleRate;

    for (let i = 0; i < frames; i += 1) {
      let weightedPower = 0;
      for (let c = 0; c < 2; c += 1) {
        const source = input.length > c ? input[c] : input[0];
        const x = source[i];
        const y = this.filters[c][1].tick(this.filters[c][0].tick(x));
        weightedPower += y * y;
      }

      // HTMLMediaElement volume is upstream of this node. Divide it out of the
      // measurement so lowering the user volume never causes the AGC to fight it.
      const volume = Math.max(this.volume, 0.001);
      this.power += weightedPower / (volume * volume);
      this.frames += 1;
      if (this.frames >= Math.round(sampleRate * 0.1)) this.analyze();

      const delta = this.desiredDb - this.gainDb;
      this.gainDb += Math.max(downStep, Math.min(upStep, delta));
      const gain = Math.pow(10, this.gainDb / 20);

      const index = this.frame % this.capacity;
      let peak = 0;
      for (let c = 0; c < 2; c += 1) {
        const source = input.length > c ? input[c] : input[0];
        const x = source[i] * gain;
        this.delay[c][index] = x;
        peak = Math.max(peak, Math.abs(x));
      }

      // Monotonic queue: O(1) look-ahead maximum without per-render allocations.
      while (this.head !== this.tail && this.peakFrames[this.head] < this.frame - this.delayFrames) {
        this.head = (this.head + 1) % this.capacity;
      }
      while (
        this.head !== this.tail
        && this.peakValues[(this.tail - 1 + this.capacity) % this.capacity] <= peak
      ) {
        this.tail = (this.tail - 1 + this.capacity) % this.capacity;
      }
      this.peakValues[this.tail] = peak;
      this.peakFrames[this.tail] = this.frame;
      this.tail = (this.tail + 1) % this.capacity;

      const limit = Math.min(1, 0.841395 / Math.max(this.peakValues[this.head], 1e-9));
      this.limiterGain = Math.min(limit, this.limiterGain + (1 - this.limiterGain) * release);

      const read = (this.frame - this.delayFrames + this.capacity) % this.capacity;
      for (let c = 0; c < output.length; c += 1) {
        output[c][i] = this.frame < this.delayFrames
          ? 0
          : this.delay[c === 0 ? 0 : 1][read] * this.limiterGain;
      }
      this.frame += 1;
    }

    return true;
  }
}

registerProcessor('hghub-balance-volume', BalanceVolumeProcessor);
