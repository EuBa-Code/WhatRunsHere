//! What it costs to run here, against what it costs to pay someone else.
//!
//! The honest comparison has two shapes and they give different answers:
//!
//! - **You already own the hardware.** The marginal cost of a token is
//!   electricity. Local nearly always wins, and the interesting number is how
//!   much per month.
//! - **You are deciding whether to buy.** Now the purchase amortises across
//!   whatever you actually run, and there is a genuine break-even volume below
//!   which an API is cheaper. Most comparisons quietly skip this and conclude
//!   that local is free.
//!
//! Both are modelled, and the break-even is reported as a monthly request
//! count, because that is the number someone can check against their own usage.

use serde::{Deserialize, Serialize};

/// Electricity draw and price.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EnergyProfile {
    /// Whole-system draw in watts while generating.
    ///
    /// Whole-system, not card-only: the CPU, memory and power supply losses are
    /// on the same bill.
    pub load_watts: f64,
    /// Price of electricity in currency units per kilowatt-hour.
    pub price_per_kwh: f64,
}

impl Default for EnergyProfile {
    /// A desktop with a mid-range discrete GPU on a typical European tariff.
    fn default() -> Self {
        Self {
            load_watts: 450.0,
            price_per_kwh: 0.25,
        }
    }
}

/// A hardware purchase being amortised, or one already made.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HardwareInvestment {
    /// Purchase price in the same currency as [`EnergyProfile::price_per_kwh`].
    pub purchase_cost: f64,
    /// Useful life over which the purchase is written down.
    pub amortisation_years: f64,
    /// Whether the hardware is already bought, making the purchase sunk.
    pub already_owned: bool,
}

impl HardwareInvestment {
    /// Hardware that is already sitting on the desk. Nothing to amortise.
    pub const OWNED: Self = Self {
        purchase_cost: 0.0,
        amortisation_years: 3.0,
        already_owned: true,
    };

    /// A prospective purchase written down over three years.
    pub const fn purchase(cost: f64) -> Self {
        Self {
            purchase_cost: cost,
            amortisation_years: 3.0,
            already_owned: false,
        }
    }

    /// The share of the purchase charged to one month.
    pub fn monthly_amortisation(&self) -> f64 {
        if self.already_owned || self.amortisation_years <= 0.0 {
            return 0.0;
        }
        self.purchase_cost / (self.amortisation_years * 12.0)
    }
}

/// Per-token pricing for a hosted model.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ApiPricing {
    /// Price per million input tokens.
    pub input_per_mtok: f64,
    /// Price per million output tokens.
    pub output_per_mtok: f64,
}

impl ApiPricing {
    /// Cost of one request with the given token counts.
    pub fn request_cost(&self, input_tokens: u32, output_tokens: u32) -> f64 {
        (f64::from(input_tokens) * self.input_per_mtok
            + f64::from(output_tokens) * self.output_per_mtok)
            / 1_000_000.0
    }
}

/// The shape of the traffic being priced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Workload {
    /// Requests served per month.
    pub requests_per_month: u64,
    /// Prompt tokens per request.
    pub input_tokens: u32,
    /// Generated tokens per request.
    pub output_tokens: u32,
}

impl Workload {
    /// A light interactive workload: a person using a chat assistant daily.
    pub const INTERACTIVE: Self = Self {
        requests_per_month: 600,
        input_tokens: 2_000,
        output_tokens: 500,
    };

    /// A coding assistant with long prompts and substantial output.
    pub const CODING: Self = Self {
        requests_per_month: 3_000,
        input_tokens: 8_000,
        output_tokens: 1_200,
    };

    /// A batch pipeline: classification or extraction over a document corpus.
    pub const BATCH: Self = Self {
        requests_per_month: 200_000,
        input_tokens: 1_500,
        output_tokens: 200,
    };

    /// Total tokens moved per month, both directions.
    pub const fn monthly_tokens(&self) -> u64 {
        self.requests_per_month * (self.input_tokens as u64 + self.output_tokens as u64)
    }
}

/// How local and hosted compare for a given workload.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CostComparison {
    /// Hours the machine spends working each month.
    pub compute_hours_per_month: f64,
    /// Whether prompt processing is included in that time.
    ///
    /// It is not when compute throughput has never been measured on this
    /// machine. Charging zero seconds for it would quietly understate both the
    /// time and the electricity, so the omission is reported rather than
    /// absorbed.
    pub prefill_included: bool,
    /// Monthly electricity.
    pub local_energy: f64,
    /// Monthly share of the hardware purchase, zero when already owned.
    pub local_amortisation: f64,
    /// Monthly cost of running locally.
    pub local_total: f64,
    /// Monthly cost of the same traffic through the API.
    pub api_total: f64,
    /// Effective local price per million tokens.
    pub local_per_mtok: f64,
    /// Effective API price per million tokens.
    pub api_per_mtok: f64,
    /// Requests per month at which local becomes the cheaper option.
    ///
    /// `None` when local is cheaper from the first request, or when it never
    /// catches up because electricity alone exceeds the API price.
    pub breakeven_requests_per_month: Option<u64>,
    /// The summary judgement.
    pub verdict: CostVerdict,
}

impl CostComparison {
    /// Monthly saving from running locally. Negative when the API is cheaper.
    pub fn monthly_saving(&self) -> f64 {
        self.api_total - self.local_total
    }
}

/// The plain-language conclusion.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub enum CostVerdict {
    /// Running locally costs less at this volume.
    LocalCheaper {
        /// Percentage saved against the API bill.
        by_percent: f64,
    },
    /// The API costs less at this volume.
    ApiCheaper {
        /// Percentage saved against running locally.
        by_percent: f64,
    },
    /// Within a few percent: the decision belongs on other grounds — privacy,
    /// latency, availability — rather than on price.
    TooCloseToCall,
}

fn verdict(local: f64, api: f64) -> CostVerdict {
    let reference = local.max(api);
    if reference <= 0.0 {
        return CostVerdict::TooCloseToCall;
    }
    // Judge the gap against the larger of the two, so neither side can be made
    // to look decisive by being close to zero.
    let relative_gap = (api - local) / reference * 100.0;
    if relative_gap.abs() <= 5.0 {
        CostVerdict::TooCloseToCall
    } else if relative_gap > 0.0 {
        // Local wins; report the saving as a share of the API bill avoided.
        CostVerdict::LocalCheaper {
            by_percent: (api - local) / api * 100.0,
        }
    } else {
        // The API wins; report the saving as a share of the local bill avoided.
        CostVerdict::ApiCheaper {
            by_percent: (local - api) / local * 100.0,
        }
    }
}

/// Seconds of compute for one request at the given speeds.
fn request_seconds(workload: &Workload, decode_tps: f64, prefill_tps: Option<f64>) -> f64 {
    let decode = if decode_tps > 0.0 {
        f64::from(workload.output_tokens) / decode_tps
    } else {
        0.0
    };
    let prefill = prefill_tps.map_or(0.0, |tps| {
        if tps > 0.0 {
            f64::from(workload.input_tokens) / tps
        } else {
            0.0
        }
    });
    prefill + decode
}

/// Compare running a model locally against paying for the same traffic.
pub fn compare(
    workload: &Workload,
    decode_tps: f64,
    prefill_tps: Option<f64>,
    energy: &EnergyProfile,
    hardware: &HardwareInvestment,
    api: &ApiPricing,
) -> CostComparison {
    let seconds_per_request = request_seconds(workload, decode_tps, prefill_tps);
    let compute_hours = seconds_per_request * (workload.requests_per_month as f64) / 3600.0;

    let energy_per_request =
        seconds_per_request / 3600.0 * energy.load_watts / 1000.0 * energy.price_per_kwh;
    let local_energy = energy_per_request * (workload.requests_per_month as f64);
    let local_amortisation = hardware.monthly_amortisation();
    let local_total = local_energy + local_amortisation;

    let api_per_request = api.request_cost(workload.input_tokens, workload.output_tokens);
    let api_total = api_per_request * (workload.requests_per_month as f64);

    // The break-even is where the fixed monthly amortisation is repaid by the
    // per-request saving. With no fixed cost there is nothing to repay.
    let margin = api_per_request - energy_per_request;
    let breakeven = if local_amortisation <= 0.0 || margin <= 0.0 {
        None
    } else {
        Some((local_amortisation / margin).ceil() as u64)
    };

    let mtok = workload.monthly_tokens() as f64 / 1_000_000.0;
    let per_mtok = |total: f64| if mtok > 0.0 { total / mtok } else { 0.0 };

    CostComparison {
        compute_hours_per_month: compute_hours,
        prefill_included: prefill_tps.is_some_and(|tps| tps > 0.0),
        local_energy,
        local_amortisation,
        local_total,
        api_total,
        local_per_mtok: per_mtok(local_total),
        api_per_mtok: per_mtok(api_total),
        breakeven_requests_per_month: breakeven,
        verdict: verdict(local_total, api_total),
    }
}

#[cfg(test)]
mod tests {
    // These assert exact zeroes and ones the code produces by construction.
    #![allow(clippy::float_cmp)]

    use super::*;

    /// Roughly what a mid-tier hosted model costs per million tokens.
    const HOSTED: ApiPricing = ApiPricing {
        input_per_mtok: 0.30,
        output_per_mtok: 1.20,
    };

    #[test]
    fn owned_hardware_makes_a_heavy_workload_dramatically_cheaper_locally() {
        let result = compare(
            &Workload::BATCH,
            120.0,
            Some(3_000.0),
            &EnergyProfile::default(),
            &HardwareInvestment::OWNED,
            &HOSTED,
        );
        assert_eq!(result.local_amortisation, 0.0);
        assert!(
            result.monthly_saving() > 0.0,
            "local {:.2} vs api {:.2}",
            result.local_total,
            result.api_total
        );
        assert!(matches!(result.verdict, CostVerdict::LocalCheaper { .. }));
        assert_eq!(result.breakeven_requests_per_month, None);
    }

    #[test]
    fn a_new_purchase_needs_real_volume_before_it_pays_for_itself() {
        let result = compare(
            &Workload::INTERACTIVE,
            120.0,
            Some(3_000.0),
            &EnergyProfile::default(),
            &HardwareInvestment::purchase(2_400.0),
            &HOSTED,
        );
        // 2400 over three years is about 67 a month; light chat use cannot
        // repay that against a bill of a few euro.
        assert!(matches!(result.verdict, CostVerdict::ApiCheaper { .. }));
        let breakeven = result
            .breakeven_requests_per_month
            .expect("a break-even exists when the margin is positive");
        assert!(
            breakeven > Workload::INTERACTIVE.requests_per_month,
            "break-even {breakeven} should exceed this workload's volume"
        );
    }

    #[test]
    fn the_reported_breakeven_is_the_actual_crossing_point() {
        let hardware = HardwareInvestment::purchase(2_400.0);
        let base = Workload::CODING;
        let at = |requests| {
            compare(
                &Workload {
                    requests_per_month: requests,
                    ..base
                },
                120.0,
                Some(3_000.0),
                &EnergyProfile::default(),
                &hardware,
                &HOSTED,
            )
        };
        let breakeven = at(base.requests_per_month)
            .breakeven_requests_per_month
            .expect("break-even exists");

        assert!(
            at(breakeven).local_total <= at(breakeven).api_total,
            "local should be cheaper at the break-even volume"
        );
        assert!(
            at(breakeven - 1).local_total > at(breakeven - 1).api_total,
            "local should still be dearer one request below it"
        );
    }

    #[test]
    fn a_slow_machine_on_expensive_power_can_lose_outright() {
        let result = compare(
            &Workload::BATCH,
            // A large model limping along on CPU.
            2.5,
            Some(40.0),
            &EnergyProfile {
                load_watts: 600.0,
                price_per_kwh: 0.45,
            },
            &HardwareInvestment::OWNED,
            &HOSTED,
        );
        assert!(matches!(result.verdict, CostVerdict::ApiCheaper { .. }));
        assert_eq!(
            result.breakeven_requests_per_month, None,
            "no volume repays a negative per-request margin"
        );
    }

    #[test]
    fn compute_hours_track_the_modelled_speeds() {
        let workload = Workload {
            requests_per_month: 3_600,
            input_tokens: 1_000,
            output_tokens: 1_000,
        };
        let result = compare(
            &workload,
            100.0,
            Some(1_000.0),
            &EnergyProfile::default(),
            &HardwareInvestment::OWNED,
            &HOSTED,
        );
        // One second of prefill plus ten of decode, 3600 times over.
        assert!((result.compute_hours_per_month - 11.0).abs() < 0.01);
        assert!(result.prefill_included);
    }

    #[test]
    fn unmeasured_prompt_processing_is_declared_rather_than_charged_at_zero() {
        let result = compare(
            &Workload::CODING,
            120.0,
            None,
            &EnergyProfile::default(),
            &HardwareInvestment::OWNED,
            &HOSTED,
        );
        assert!(
            !result.prefill_included,
            "a cost built on an unknown must say so"
        );
        assert!(result.compute_hours_per_month > 0.0);
    }

    #[test]
    fn per_million_token_prices_are_comparable_across_both_sides() {
        let result = compare(
            &Workload::CODING,
            120.0,
            Some(3_000.0),
            &EnergyProfile::default(),
            &HardwareInvestment::OWNED,
            &HOSTED,
        );
        assert!(result.api_per_mtok > 0.0);
        assert!(result.local_per_mtok > 0.0);
        assert!(
            result.local_per_mtok < result.api_per_mtok,
            "local {:.3} vs api {:.3} per Mtok",
            result.local_per_mtok,
            result.api_per_mtok
        );
    }

    #[test]
    fn workload_token_counts_are_what_they_claim() {
        assert_eq!(Workload::INTERACTIVE.monthly_tokens(), 600 * (2_000 + 500));
    }
}
