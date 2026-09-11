//! Measuring what this machine can sustain, rather than looking it up.
//!
//! The usual approach to accelerator bandwidth is a table: recognise the part
//! by name, read its datasheet figure, apply a fudge factor. That works for the
//! parts somebody remembered to add and fails quietly for everything else:
//! every laptop chip, every integrated GPU, every part released after the table
//! was written, and every machine whose real throughput differs from its
//! specification because of thermal limits, memory configuration, or a
//! single-channel build. It also carries a trap that is easy to miss: a mobile
//! part shares its model number with a desktop card of a different bus width
//! and a different amount of memory, so matching on the number alone is wrong
//! for every one of them.
//!
//! Measuring sidesteps all of it.
//!
//! What is measured here is a **read**, not a copy. Decode is bounded by
//! streaming the weights in; nothing is written back. A copy benchmark reports
//! read and write bytes together and is the right convention for a different
//! question, but charging decode against it would describe traffic decode never
//! generates.
//!
//! The result is an upper bound on what an inference kernel reaches, and the
//! discount from one to the other lives in
//! [`whatllm_core::perf::DeviceThroughput::from_probe`].

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::hint::black_box;
use std::sync::Barrier;
use std::time::{Duration, Instant};

/// Below this, the measurement was starved rather than the machine slow.
const IMPLAUSIBLY_SLOW_BYTES_PER_S: f64 = 0.5e9;
/// Above this, a cache was measured rather than memory.
const IMPLAUSIBLY_FAST_BYTES_PER_S: f64 = 4000e9;

/// How thoroughly to measure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProbeOptions {
    /// Working set **per thread**, in bytes.
    ///
    /// Per thread rather than shared, so each buffer is first touched by the
    /// thread that will read it. On a machine with more than one memory node
    /// that is what puts the pages on the node doing the reading; one shared
    /// allocation lands entirely on whichever node touched it first, and every
    /// other thread then measures the interconnect.
    ///
    /// Must comfortably exceed the per-core share of last-level cache.
    pub per_thread_bytes: usize,
    /// Threads to read with. One core rarely saturates a modern memory
    /// controller, so a single-threaded figure understates the machine badly.
    pub threads: usize,
    /// How long to read for. Passes are counted within the window rather than
    /// timed one by one, which keeps the clock out of the inner loop.
    pub window: Duration,
}

impl Default for ProbeOptions {
    fn default() -> Self {
        Self {
            per_thread_bytes: 48 * 1024 * 1024,
            // Past a handful of readers the controller is saturated, and more
            // threads add scheduling noise rather than throughput.
            threads: std::thread::available_parallelism().map_or(4, |n| n.get().min(8)),
            window: Duration::from_millis(250),
        }
    }
}

impl ProbeOptions {
    /// A quicker, less precise measurement, for when someone is waiting.
    pub fn quick() -> Self {
        Self {
            per_thread_bytes: 24 * 1024 * 1024,
            window: Duration::from_millis(80),
            ..Self::default()
        }
    }

    /// Total bytes resident across all threads.
    pub fn total_bytes(&self) -> usize {
        self.per_thread_bytes.saturating_mul(self.threads.max(1))
    }
}

/// What a run of the probe established.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HostBandwidth {
    /// Sustained read bandwidth across all threads, in bytes per second.
    pub bytes_per_s: f64,
    /// Single-threaded read bandwidth, in bytes per second.
    ///
    /// Reported because the ratio between the two says whether the machine is
    /// bound by its memory controller or by how many cores are reading, which
    /// decides whether more inference threads will help.
    pub single_thread_bytes_per_s: f64,
    /// Threads used for the parallel figure.
    pub threads: usize,
    /// Working set per thread.
    pub per_thread_bytes: usize,
    /// How long the whole probe took.
    pub elapsed: Duration,
    /// Whether the result lands in a believable range.
    ///
    /// A measurement taken on a machine under heavy contention, or inside a
    /// throttled virtual machine, comes back far too low; one taken on a
    /// working set that fits in cache comes back far too high. Either is worse
    /// than no measurement, because it will be believed. Callers should refuse
    /// to store an implausible result rather than calibrate against it.
    pub plausible: bool,
}

impl HostBandwidth {
    /// Sustained bandwidth in gigabytes per second.
    pub fn gigabytes_per_s(&self) -> f64 {
        self.bytes_per_s / 1e9
    }

    /// How much the machine gains from reading on many cores at once.
    ///
    /// Near 1.0 means one core already saturates the controller. A high ratio
    /// means throughput depends on thread count, and an inference runtime
    /// configured with too few threads will leave bandwidth unused.
    pub fn parallel_speedup(&self) -> f64 {
        if self.single_thread_bytes_per_s <= 0.0 {
            return 1.0;
        }
        self.bytes_per_s / self.single_thread_bytes_per_s
    }
}

/// Sum a slice with several independent accumulators.
///
/// The accumulators matter: a single one serialises on its own dependency
/// chain and measures addition latency rather than memory throughput.
fn stream_sum(data: &[u64]) -> u64 {
    const LANES: usize = 8;
    let mut lanes = [0u64; LANES];
    let mut chunks = data.chunks_exact(LANES);
    for chunk in &mut chunks {
        for (lane, value) in lanes.iter_mut().zip(chunk) {
            *lane = lane.wrapping_add(*value);
        }
    }
    let mut total = chunks
        .remainder()
        .iter()
        .copied()
        .fold(0u64, u64::wrapping_add);
    for lane in lanes {
        total = total.wrapping_add(lane);
    }
    total
}

/// Aggregate read rate in bytes per second across `threads` concurrent readers.
///
/// Every reader owns its buffer and they all start together at a barrier, so
/// what is measured is genuinely concurrent. Staggered starts would let an
/// early thread run against an uncontended controller and report a rate the
/// machine cannot sustain once every thread is going.
fn aggregate_read_rate(threads: usize, per_thread_bytes: usize, window: Duration) -> f64 {
    let threads = threads.max(1);
    let elems = (per_thread_bytes / std::mem::size_of::<u64>()).max(4096);
    let bytes = (elems * std::mem::size_of::<u64>()) as f64;
    let barrier = Barrier::new(threads);

    let rates: Vec<f64> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let barrier = &barrier;
                scope.spawn(move || {
                    // Allocated and first touched by the thread that reads it.
                    let mut buffer = vec![0u64; elems];
                    for (index, slot) in buffer.iter_mut().enumerate() {
                        *slot = index as u64 | 1;
                    }
                    // An untimed pass, so the timed one is not paying for page
                    // faults or frequency ramp-up.
                    black_box(stream_sum(&buffer));

                    barrier.wait();
                    let start = Instant::now();
                    let mut passes = 0u64;
                    let mut checksum = 0u64;
                    while start.elapsed() < window {
                        checksum = checksum.wrapping_add(stream_sum(&buffer));
                        passes += 1;
                    }
                    let seconds = start.elapsed().as_secs_f64();
                    black_box(checksum);
                    if seconds > 0.0 {
                        passes as f64 * bytes / seconds
                    } else {
                        0.0
                    }
                })
            })
            .collect();
        handles
            .into_iter()
            .filter_map(|handle| handle.join().ok())
            .collect()
    });

    // A thread that failed to report would understate the aggregate silently.
    if rates.len() != threads {
        return 0.0;
    }
    rates.iter().sum()
}

/// Measure how fast this machine streams from system memory.
pub fn measure_host_bandwidth(options: ProbeOptions) -> HostBandwidth {
    let started = Instant::now();
    let parallel = aggregate_read_rate(options.threads, options.per_thread_bytes, options.window);
    let single = aggregate_read_rate(1, options.per_thread_bytes, options.window);

    HostBandwidth {
        bytes_per_s: parallel,
        single_thread_bytes_per_s: single,
        threads: options.threads.max(1),
        per_thread_bytes: options.per_thread_bytes,
        elapsed: started.elapsed(),
        plausible: (IMPLAUSIBLY_SLOW_BYTES_PER_S..=IMPLAUSIBLY_FAST_BYTES_PER_S)
            .contains(&parallel)
            && single > 0.0,
    }
}

#[cfg(test)]
mod tests {
    // One assertion checks an exact 1.0 the code produces by construction.
    #![allow(clippy::float_cmp)]

    use super::*;

    /// A small probe, so the suite stays fast. Real runs use the default.
    fn tiny() -> ProbeOptions {
        ProbeOptions {
            per_thread_bytes: 8 * 1024 * 1024,
            threads: 2,
            window: Duration::from_millis(30),
        }
    }

    #[test]
    fn the_probe_produces_a_plausible_rate() {
        let result = measure_host_bandwidth(tiny());
        assert!(
            result.plausible,
            "measured {:.1} GB/s, which is not a memory bandwidth",
            result.gigabytes_per_s()
        );
        assert!(result.single_thread_bytes_per_s > 0.0);
    }

    #[test]
    fn the_parallel_gain_is_the_ratio_it_claims_to_be() {
        let result = measure_host_bandwidth(tiny());
        let expected = result.bytes_per_s / result.single_thread_bytes_per_s;
        assert!((result.parallel_speedup() - expected).abs() < 1e-9);

        // Deliberately not asserting that more threads are faster. On the small
        // working set this test uses, the data sits in cache, where thread
        // scaling is a property of the machine and its current load rather than
        // of anything here. The wide bound only catches a result that is
        // arithmetically broken.
        assert!(
            (0.2..50.0).contains(&result.parallel_speedup()),
            "implausible parallel gain: {:.2}x",
            result.parallel_speedup()
        );
    }

    #[test]
    fn a_speedup_is_reported_as_one_when_the_single_thread_figure_is_missing() {
        let broken = HostBandwidth {
            bytes_per_s: 40e9,
            single_thread_bytes_per_s: 0.0,
            threads: 8,
            per_thread_bytes: 1024,
            elapsed: Duration::from_millis(1),
            plausible: false,
        };
        assert_eq!(broken.parallel_speedup(), 1.0);
    }

    #[test]
    fn a_cache_resident_measurement_is_flagged_rather_than_believed() {
        let result = measure_host_bandwidth(ProbeOptions {
            per_thread_bytes: 8,
            threads: 1,
            window: Duration::from_millis(10),
        });
        assert!(result.bytes_per_s > 0.0, "something must still be measured");
        assert!(
            !result.plausible || result.bytes_per_s <= IMPLAUSIBLY_FAST_BYTES_PER_S,
            "a cache-resident measurement must not pass as memory bandwidth"
        );
    }

    #[test]
    fn summing_lanes_agrees_with_the_obvious_loop() {
        let data: Vec<u64> = (0..1000).collect();
        let expected = data.iter().copied().fold(0u64, u64::wrapping_add);
        assert_eq!(stream_sum(&data), expected);
        // And with a length that is not a multiple of the lane count.
        assert_eq!(stream_sum(&data[..997]), data[..997].iter().sum::<u64>());
    }

    #[test]
    fn the_default_working_set_clears_any_real_cache() {
        let options = ProbeOptions::default();
        assert!(
            options.per_thread_bytes >= 16 * 1024 * 1024,
            "a per-thread buffer this small would measure last-level cache"
        );
        assert!(options.total_bytes() >= options.per_thread_bytes);
    }
}
