use anyhow::{Context, Result};
use ash::vk;
use std::cell::RefCell;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

const DEFAULT_CSV_FLUSH_INTERVAL_FRAMES: u64 = 30;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuProfilerConfig {
    pub enabled: bool,
    pub csv_path: Option<PathBuf>,
    pub csv_flush_interval_frames: u64,
}

impl GpuProfilerConfig {
    pub fn from_env() -> Self {
        let profiler = std::env::var("REVOLUMETRIC_GPU_PROFILER").ok();
        let csv = std::env::var("REVOLUMETRIC_GPU_PROFILE_CSV").ok();
        let flush_interval = std::env::var("REVOLUMETRIC_GPU_PROFILE_CSV_FLUSH_INTERVAL").ok();
        Self::from_values(
            profiler.as_deref(),
            csv.as_deref(),
            flush_interval.as_deref(),
            cfg!(debug_assertions),
        )
    }

    pub fn from_values(
        profiler: Option<&str>,
        csv_path: Option<&str>,
        csv_flush_interval: Option<&str>,
        debug_default: bool,
    ) -> Self {
        let csv_path = csv_path
            .filter(|path| !path.trim().is_empty())
            .map(PathBuf::from);
        let enabled = match profiler.map(str::trim) {
            Some(value)
                if value == "1"
                    || value.eq_ignore_ascii_case("true")
                    || value.eq_ignore_ascii_case("on") =>
            {
                true
            }
            Some(value)
                if value == "0"
                    || value.eq_ignore_ascii_case("false")
                    || value.eq_ignore_ascii_case("off") =>
            {
                false
            }
            Some(_) => debug_default || csv_path.is_some(),
            None => debug_default || csv_path.is_some(),
        };

        let csv_flush_interval_frames = csv_flush_interval
            .and_then(|value| value.trim().parse::<u64>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_CSV_FLUSH_INTERVAL_FRAMES);

        Self {
            enabled,
            csv_path,
            csv_flush_interval_frames,
        }
    }
}

#[repr(usize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuProfileScope {
    PrimaryRay = 0,
    RcClear = 1,
    RcTraceC0 = 2,
    RcTraceC1 = 3,
    RcTraceC2 = 4,
    RcMergeC2ToC1 = 5,
    RcMergeC1ToC0 = 6,
    Lighting = 7,
    BlitToSwapchain = 8,
}

impl GpuProfileScope {
    pub const COUNT: usize = 9;
    pub const ALL: [Self; Self::COUNT] = [
        Self::PrimaryRay,
        Self::RcClear,
        Self::RcTraceC0,
        Self::RcTraceC1,
        Self::RcTraceC2,
        Self::RcMergeC2ToC1,
        Self::RcMergeC1ToC0,
        Self::Lighting,
        Self::BlitToSwapchain,
    ];

    pub fn log_name(self) -> &'static str {
        match self {
            Self::PrimaryRay => "PrimaryRay",
            Self::RcClear => "RCClear",
            Self::RcTraceC0 => "RC-C0",
            Self::RcTraceC1 => "RC-C1",
            Self::RcTraceC2 => "RC-C2",
            Self::RcMergeC2ToC1 => "Merge C2->C1",
            Self::RcMergeC1ToC0 => "Merge C1->C0",
            Self::Lighting => "Lighting",
            Self::BlitToSwapchain => "Blit",
        }
    }

    pub fn csv_column(self) -> &'static str {
        match self {
            Self::PrimaryRay => "primary_ray_ms",
            Self::RcClear => "rc_clear_ms",
            Self::RcTraceC0 => "rc_trace_c0_ms",
            Self::RcTraceC1 => "rc_trace_c1_ms",
            Self::RcTraceC2 => "rc_trace_c2_ms",
            Self::RcMergeC2ToC1 => "rc_merge_c2_to_c1_ms",
            Self::RcMergeC1ToC0 => "rc_merge_c1_to_c0_ms",
            Self::Lighting => "lighting_ms",
            Self::BlitToSwapchain => "blit_to_swapchain_ms",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct QueryLayout {
    frame_slots: usize,
}

impl QueryLayout {
    pub fn new(frame_slots: usize) -> Self {
        Self { frame_slots }
    }

    pub fn query_count(self) -> u32 {
        (self.frame_slots * GpuProfileScope::COUNT * 2) as u32
    }

    pub fn begin_query(self, frame_slot: usize, scope: GpuProfileScope) -> u32 {
        self.base_query(frame_slot, scope)
    }

    pub fn end_query(self, frame_slot: usize, scope: GpuProfileScope) -> u32 {
        self.base_query(frame_slot, scope) + 1
    }

    pub fn frame_first_query(self, frame_slot: usize) -> u32 {
        (frame_slot * GpuProfileScope::COUNT * 2) as u32
    }

    pub fn frame_query_count(self) -> u32 {
        (GpuProfileScope::COUNT * 2) as u32
    }

    fn base_query(self, frame_slot: usize, scope: GpuProfileScope) -> u32 {
        self.frame_first_query(frame_slot) + (scope as u32 * 2)
    }
}

pub fn timestamp_delta_ms(begin: u64, end: u64, timestamp_period_ns: f64) -> Option<f64> {
    (end >= begin).then(|| (end - begin) as f64 * timestamp_period_ns / 1_000_000.0)
}

#[derive(Debug, Clone)]
pub struct GpuFrameTimings {
    pub frame_index: u64,
    pub scope_ms: [f64; GpuProfileScope::COUNT],
}

impl GpuFrameTimings {
    pub fn new(frame_index: u64, scope_ms: [f64; GpuProfileScope::COUNT]) -> Self {
        Self {
            frame_index,
            scope_ms,
        }
    }

    pub fn total_ms(&self) -> f64 {
        self.scope_ms.iter().sum()
    }
}

#[derive(Debug, Clone)]
pub struct GpuTimingSummary {
    pub frame_count: u64,
    pub average_ms: [f64; GpuProfileScope::COUNT],
}

impl GpuTimingSummary {
    pub fn total_ms(&self) -> f64 {
        self.average_ms.iter().sum()
    }

    pub fn format_log_line(&self) -> String {
        let mut parts = Vec::with_capacity(GpuProfileScope::COUNT + 2);
        for scope in GpuProfileScope::ALL {
            parts.push(format!(
                "{}: {:.2}ms",
                scope.log_name(),
                self.average_ms[scope as usize]
            ));
        }
        parts.push(format!("Total: {:.2}ms", self.total_ms()));
        format!("[GPU] {}", parts.join(" | "))
    }
}

#[derive(Debug, Clone)]
pub struct SummaryAccumulator {
    interval: u64,
    frames: u64,
    totals: [f64; GpuProfileScope::COUNT],
}

impl SummaryAccumulator {
    pub fn new(interval: u64) -> Self {
        Self {
            interval: interval.max(1),
            frames: 0,
            totals: [0.0; GpuProfileScope::COUNT],
        }
    }

    pub fn push(&mut self, timings: &GpuFrameTimings) -> Option<GpuTimingSummary> {
        self.frames += 1;
        for (total, value) in self.totals.iter_mut().zip(timings.scope_ms) {
            *total += value;
        }

        if self.frames < self.interval {
            return None;
        }

        let frame_count = self.frames;
        let mut average_ms = [0.0; GpuProfileScope::COUNT];
        for (average, total) in average_ms.iter_mut().zip(self.totals) {
            *average = total / frame_count as f64;
        }
        self.frames = 0;
        self.totals = [0.0; GpuProfileScope::COUNT];

        Some(GpuTimingSummary {
            frame_count,
            average_ms,
        })
    }
}

pub fn csv_header() -> String {
    let mut columns = Vec::with_capacity(GpuProfileScope::COUNT + 2);
    columns.push("frame");
    columns.extend(GpuProfileScope::ALL.iter().map(|scope| scope.csv_column()));
    columns.push("total_ms");
    columns.join(",")
}

pub fn csv_row(timings: &GpuFrameTimings) -> String {
    let mut columns = Vec::with_capacity(GpuProfileScope::COUNT + 2);
    columns.push(timings.frame_index.to_string());
    columns.extend(timings.scope_ms.iter().map(|value| format!("{value:.4}")));
    columns.push(format!("{:.4}", timings.total_ms()));
    columns.join(",")
}

pub fn timestamp_support_available(timestamp_valid_bits: u32, timestamp_period_ns: f32) -> bool {
    timestamp_valid_bits > 0 && timestamp_period_ns > 0.0
}

pub struct GpuProfiler {
    query_pool: vk::QueryPool,
    layout: QueryLayout,
    timestamp_period_ns: f64,
    frame_slots: RefCell<Vec<FrameSlotState>>,
    summary: SummaryAccumulator,
    csv: Option<CsvWriter>,
}

#[derive(Debug, Clone, Copy)]
struct FrameSlotState {
    initialized: bool,
    frame_index: u64,
    written_scopes: [bool; GpuProfileScope::COUNT],
}

struct CsvWriter {
    writer: BufWriter<File>,
    disabled: bool,
    flush_interval_frames: u64,
    rows_since_flush: u64,
}

fn take_pending_frame_slot(
    frame_slots: &mut [FrameSlotState],
    frame_slot: usize,
) -> Option<FrameSlotState> {
    let state = frame_slots.get_mut(frame_slot)?;
    if !state.initialized {
        return None;
    }
    state.initialized = false;
    Some(*state)
}

impl GpuProfiler {
    pub fn new(
        device: &ash::Device,
        timestamp_period_ns: f32,
        timestamp_valid_bits: u32,
        frame_slot_count: usize,
        config: GpuProfilerConfig,
    ) -> Result<Option<Self>> {
        if !config.enabled {
            return Ok(None);
        }
        if !timestamp_support_available(timestamp_valid_bits, timestamp_period_ns) {
            tracing::warn!(
                timestamp_valid_bits,
                timestamp_period_ns,
                "GPU timestamp profiling disabled because the graphics queue does not support timestamps"
            );
            return Ok(None);
        }

        let layout = QueryLayout::new(frame_slot_count);
        let create_info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(layout.query_count());
        let query_pool = unsafe { device.create_query_pool(&create_info, None) }
            .context("failed to create GPU timestamp query pool")?;
        let csv = CsvWriter::new(config.csv_path, config.csv_flush_interval_frames);

        tracing::info!(
            frame_slot_count,
            timestamp_period_ns,
            "enabled GPU timestamp profiler"
        );

        Ok(Some(Self {
            query_pool,
            layout,
            timestamp_period_ns: timestamp_period_ns as f64,
            frame_slots: RefCell::new(vec![
                FrameSlotState {
                    initialized: false,
                    frame_index: 0,
                    written_scopes: [false; GpuProfileScope::COUNT],
                };
                frame_slot_count
            ]),
            summary: SummaryAccumulator::new(60),
            csv,
        }))
    }

    pub fn begin_frame(
        &mut self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        frame_slot: usize,
        frame_index: u64,
    ) {
        self.read_finished_frame(device, frame_slot);
        unsafe {
            device.cmd_reset_query_pool(
                command_buffer,
                self.query_pool,
                self.layout.frame_first_query(frame_slot),
                self.layout.frame_query_count(),
            );
        }
        if let Some(state) = self.frame_slots.borrow_mut().get_mut(frame_slot) {
            state.initialized = true;
            state.frame_index = frame_index;
            state.written_scopes = [false; GpuProfileScope::COUNT];
        }
    }

    pub fn begin_scope(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        frame_slot: usize,
        scope: GpuProfileScope,
    ) {
        if let Some(state) = self.frame_slots.borrow_mut().get_mut(frame_slot) {
            state.written_scopes[scope as usize] = true;
        }
        unsafe {
            device.cmd_write_timestamp(
                command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                self.query_pool,
                self.layout.begin_query(frame_slot, scope),
            );
        }
    }

    pub fn end_scope(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        frame_slot: usize,
        scope: GpuProfileScope,
    ) {
        unsafe {
            device.cmd_write_timestamp(
                command_buffer,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.query_pool,
                self.layout.end_query(frame_slot, scope),
            );
        }
    }

    pub fn flush_pending_frames(&mut self, device: &ash::Device) {
        let frame_slot_count = self.frame_slots.borrow().len();
        for frame_slot in 0..frame_slot_count {
            self.read_finished_frame(device, frame_slot);
        }
        if let Some(csv) = &mut self.csv {
            csv.flush();
        }
    }

    pub fn destroy(mut self, device: &ash::Device) {
        self.flush_pending_frames(device);
        unsafe { device.destroy_query_pool(self.query_pool, None) };
    }

    fn read_finished_frame(&mut self, device: &ash::Device, frame_slot: usize) {
        let state = {
            let mut frame_slots = self.frame_slots.borrow_mut();
            take_pending_frame_slot(frame_slots.as_mut_slice(), frame_slot)
        };
        let Some(state) = state else { return };

        let mut scope_ms = [0.0; GpuProfileScope::COUNT];
        let mut valid_scope_count = 0;
        for scope in GpuProfileScope::ALL {
            let scope_index = scope as usize;
            if !state.written_scopes[scope_index] {
                continue;
            }

            let mut raw_results = [0_u64; 2];
            let result = unsafe {
                device.get_query_pool_results(
                    self.query_pool,
                    self.layout.begin_query(frame_slot, scope),
                    &mut raw_results,
                    vk::QueryResultFlags::TYPE_64,
                )
            };
            if let Err(error) = result {
                tracing::warn!(
                    ?error,
                    scope = scope.log_name(),
                    "failed to read GPU timestamp query results"
                );
                continue;
            }

            if let Some(ms) =
                timestamp_delta_ms(raw_results[0], raw_results[1], self.timestamp_period_ns)
            {
                scope_ms[scope_index] = ms;
                valid_scope_count += 1;
            }
        }

        if valid_scope_count == 0 {
            return;
        }

        let timings = GpuFrameTimings::new(state.frame_index, scope_ms);
        if let Some(csv) = &mut self.csv {
            csv.write_row(&timings);
        }
        if let Some(summary) = self.summary.push(&timings) {
            tracing::info!("{}", summary.format_log_line());
        }
    }
}

impl CsvWriter {
    fn new(path: Option<PathBuf>, flush_interval_frames: u64) -> Option<Self> {
        let path = path?;
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            if let Err(error) = fs::create_dir_all(parent) {
                tracing::warn!(?path, ?error, "failed to create GPU profiler CSV directory");
                return None;
            }
        }

        let file = match File::create(&path) {
            Ok(file) => file,
            Err(error) => {
                tracing::warn!(?path, ?error, "failed to create GPU profiler CSV file");
                return None;
            }
        };
        let mut writer = BufWriter::new(file);
        if let Err(error) = writeln!(writer, "{}", csv_header()).and_then(|_| writer.flush()) {
            tracing::warn!(?path, ?error, "failed to write GPU profiler CSV header");
            return None;
        }
        Some(Self {
            writer,
            disabled: false,
            flush_interval_frames: flush_interval_frames.max(1),
            rows_since_flush: 0,
        })
    }

    fn write_row(&mut self, timings: &GpuFrameTimings) {
        if self.disabled {
            return;
        }
        if let Err(error) = writeln!(self.writer, "{}", csv_row(timings)) {
            tracing::warn!(
                ?error,
                "failed to write GPU profiler CSV row; disabling CSV output"
            );
            self.disabled = true;
            return;
        }

        self.rows_since_flush += 1;
        if self.rows_since_flush >= self.flush_interval_frames {
            self.flush();
        }
    }

    fn flush(&mut self) {
        if self.disabled || self.rows_since_flush == 0 {
            return;
        }
        if let Err(error) = self.writer.flush() {
            tracing::warn!(
                ?error,
                "failed to flush GPU profiler CSV; disabling CSV output"
            );
            self.disabled = true;
        } else {
            self.rows_since_flush = 0;
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_to_debug_setting() {
        let enabled = GpuProfilerConfig::from_values(None, None, None, true);
        let disabled = GpuProfilerConfig::from_values(None, None, None, false);

        assert!(enabled.enabled);
        assert!(!disabled.enabled);
        assert!(enabled.csv_path.is_none());
    }

    #[test]
    fn config_env_override_wins_over_defaults_and_csv() {
        let forced_on = GpuProfilerConfig::from_values(Some("1"), None, None, false);
        let forced_off =
            GpuProfilerConfig::from_values(Some("0"), Some("target/gpu.csv"), None, true);

        assert!(forced_on.enabled);
        assert!(!forced_off.enabled);
        assert_eq!(
            forced_off.csv_path.as_deref(),
            Some(std::path::Path::new("target/gpu.csv"))
        );
    }

    #[test]
    fn config_env_override_is_case_insensitive() {
        assert!(GpuProfilerConfig::from_values(Some("True"), None, None, false).enabled);
        assert!(GpuProfilerConfig::from_values(Some("On"), None, None, false).enabled);
        assert!(
            !GpuProfilerConfig::from_values(Some("False"), Some("target/gpu.csv"), None, true)
                .enabled
        );
        assert!(
            !GpuProfilerConfig::from_values(Some("Off"), Some("target/gpu.csv"), None, true)
                .enabled
        );
    }

    #[test]
    fn config_parses_csv_flush_interval_with_safe_default() {
        let defaulted = GpuProfilerConfig::from_values(None, Some("target/gpu.csv"), None, false);
        let custom = GpuProfilerConfig::from_values(None, Some("target/gpu.csv"), Some("4"), false);
        let invalid =
            GpuProfilerConfig::from_values(None, Some("target/gpu.csv"), Some("0"), false);

        assert_eq!(
            defaulted.csv_flush_interval_frames,
            DEFAULT_CSV_FLUSH_INTERVAL_FRAMES
        );
        assert_eq!(custom.csv_flush_interval_frames, 4);
        assert_eq!(
            invalid.csv_flush_interval_frames,
            DEFAULT_CSV_FLUSH_INTERVAL_FRAMES
        );
    }

    #[test]
    fn csv_path_enables_profiler_when_no_explicit_override() {
        let config = GpuProfilerConfig::from_values(None, Some("target/gpu.csv"), None, false);

        assert!(config.enabled);
        assert_eq!(
            config.csv_path.as_deref(),
            Some(std::path::Path::new("target/gpu.csv"))
        );
    }

    #[test]
    fn scope_names_and_csv_columns_are_stable() {
        let names: Vec<_> = GpuProfileScope::ALL
            .iter()
            .map(|scope| scope.log_name())
            .collect();
        let columns: Vec<_> = GpuProfileScope::ALL
            .iter()
            .map(|scope| scope.csv_column())
            .collect();

        assert_eq!(GpuProfileScope::COUNT, 9);
        assert_eq!(names[0], "PrimaryRay");
        assert_eq!(names[5], "Merge C2->C1");
        assert_eq!(names[6], "Merge C1->C0");
        assert_eq!(names[8], "Blit");
        assert_eq!(columns[0], "primary_ray_ms");
        assert_eq!(columns[8], "blit_to_swapchain_ms");
    }

    #[test]
    fn query_layout_allocates_two_queries_per_scope_per_frame_slot() {
        let layout = QueryLayout::new(2);

        assert_eq!(
            layout.query_count(),
            (GpuProfileScope::COUNT * 2 * 2) as u32
        );
        assert_eq!(layout.begin_query(0, GpuProfileScope::PrimaryRay), 0);
        assert_eq!(layout.end_query(0, GpuProfileScope::PrimaryRay), 1);
        assert_eq!(layout.begin_query(0, GpuProfileScope::RcClear), 2);
        assert_eq!(
            layout.begin_query(1, GpuProfileScope::PrimaryRay),
            (GpuProfileScope::COUNT * 2) as u32
        );
    }

    #[test]
    fn timestamp_ticks_convert_to_milliseconds() {
        let ms = timestamp_delta_ms(1_000, 3_500, 2.0).unwrap();

        assert_eq!(ms, 0.005);
        assert!(timestamp_delta_ms(10, 5, 1.0).is_none());
    }

    #[test]
    fn summary_accumulates_and_resets_on_interval() {
        let mut accumulator = SummaryAccumulator::new(2);
        let mut first = [0.0; GpuProfileScope::COUNT];
        first[GpuProfileScope::PrimaryRay as usize] = 1.0;
        first[GpuProfileScope::Lighting as usize] = 3.0;
        let mut second = [0.0; GpuProfileScope::COUNT];
        second[GpuProfileScope::PrimaryRay as usize] = 3.0;
        second[GpuProfileScope::Lighting as usize] = 5.0;

        assert!(accumulator.push(&GpuFrameTimings::new(1, first)).is_none());
        let summary = accumulator.push(&GpuFrameTimings::new(2, second)).unwrap();

        assert_eq!(summary.frame_count, 2);
        assert_eq!(
            summary.average_ms[GpuProfileScope::PrimaryRay as usize],
            2.0
        );
        assert_eq!(summary.average_ms[GpuProfileScope::Lighting as usize], 4.0);
        assert_eq!(summary.total_ms(), 6.0);
        assert!(accumulator.push(&GpuFrameTimings::new(3, first)).is_none());
    }

    #[test]
    fn timestamp_support_requires_valid_bits_and_positive_period() {
        assert!(timestamp_support_available(64, 1.0));
        assert!(!timestamp_support_available(0, 1.0));
        assert!(!timestamp_support_available(64, 0.0));
    }
    #[test]
    fn csv_writer_flushes_on_configured_interval() {
        let mut path = std::env::current_dir().unwrap();
        path.push("target/gpu-profiler-flush-test.csv");
        let _ = std::fs::remove_file(&path);
        let mut writer = CsvWriter::new(Some(path.clone()), 2).unwrap();
        let mut timings = [0.0; GpuProfileScope::COUNT];
        timings[GpuProfileScope::PrimaryRay as usize] = 2.5;

        writer.write_row(&GpuFrameTimings::new(7, timings));
        let contents = std::fs::read_to_string(&path).unwrap();
        assert!(contents.contains("frame,primary_ray_ms"));
        assert!(!contents.contains("7,2.5000"));

        writer.write_row(&GpuFrameTimings::new(8, timings));

        let contents = std::fs::read_to_string(&path).unwrap();
        assert!(contents.contains("frame,primary_ray_ms"));
        assert!(contents.contains("7,2.5000"));
        assert!(contents.contains("8,2.5000"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn csv_writer_flush_interval_can_preserve_live_inspection() {
        let mut path = std::env::current_dir().unwrap();
        path.push("target/gpu-profiler-live-flush-test.csv");
        let _ = std::fs::remove_file(&path);
        let mut writer = CsvWriter::new(Some(path.clone()), 1).unwrap();
        let timings = [0.0; GpuProfileScope::COUNT];

        writer.write_row(&GpuFrameTimings::new(9, timings));

        let contents = std::fs::read_to_string(&path).unwrap();
        assert!(contents.contains("9,"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn pending_frame_slot_take_marks_slot_drained() {
        let mut states = vec![
            FrameSlotState {
                initialized: true,
                frame_index: 10,
                written_scopes: [true; GpuProfileScope::COUNT],
            },
            FrameSlotState {
                initialized: false,
                frame_index: 0,
                written_scopes: [false; GpuProfileScope::COUNT],
            },
        ];

        let state = take_pending_frame_slot(&mut states, 0).unwrap();

        assert_eq!(state.frame_index, 10);
        assert!(!states[0].initialized);
        assert!(take_pending_frame_slot(&mut states, 0).is_none());
        assert!(take_pending_frame_slot(&mut states, 1).is_none());
    }

    #[test]
    fn csv_header_and_rows_match_scope_order() {
        let mut timings = [0.0; GpuProfileScope::COUNT];
        timings[GpuProfileScope::PrimaryRay as usize] = 1.25;
        timings[GpuProfileScope::BlitToSwapchain as usize] = 0.5;
        let frame = GpuFrameTimings::new(42, timings);

        assert_eq!(
            csv_header(),
            "frame,primary_ray_ms,rc_clear_ms,rc_trace_c0_ms,rc_trace_c1_ms,rc_trace_c2_ms,rc_merge_c2_to_c1_ms,rc_merge_c1_to_c0_ms,lighting_ms,blit_to_swapchain_ms,total_ms"
        );
        assert_eq!(
            csv_row(&frame),
            "42,1.2500,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.5000,1.7500"
        );
    }
}
