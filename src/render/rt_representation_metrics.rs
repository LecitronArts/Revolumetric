use std::fmt::Write;

pub const RT_REPRESENTATION_METRICS_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtRepresentationKind {
    ReferenceDda,
    CompactExact,
    HotOmm,
    CompactGreedy,
    HotInterface,
    MegaGeometry,
}

impl RtRepresentationKind {
    pub const ALL: [Self; 6] = [
        Self::ReferenceDda,
        Self::CompactExact,
        Self::HotOmm,
        Self::CompactGreedy,
        Self::HotInterface,
        Self::MegaGeometry,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            Self::ReferenceDda => "reference_dda",
            Self::CompactExact => "compact_exact",
            Self::HotOmm => "hot_omm",
            Self::CompactGreedy => "compact_greedy",
            Self::HotInterface => "hot_interface",
            Self::MegaGeometry => "mega_geometry",
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RtRepresentationTimings {
    pub surface_generation_ms: Option<f64>,
    pub omm_build_ms: Option<f64>,
    pub blas_build_ms: Option<f64>,
    pub blas_update_ms: Option<f64>,
    pub tlas_ms: Option<f64>,
    pub primary_trace_ms: Option<f64>,
    pub shadow_trace_ms: Option<f64>,
    pub gi_trace_ms: Option<f64>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RtRepresentationMemory {
    pub persistent_bytes: u64,
    pub scratch_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RtRepresentationMetrics {
    pub representation: RtRepresentationKind,
    pub page_count: u64,
    pub exposed_face_count: u64,
    pub candidate_primitive_count: u64,
    pub transparent_or_degenerate_primitive_count: u64,
    pub timings: RtRepresentationTimings,
    pub memory: RtRepresentationMemory,
}

impl RtRepresentationMetrics {
    pub fn empty(representation: RtRepresentationKind) -> Self {
        Self {
            representation,
            page_count: 0,
            exposed_face_count: 0,
            candidate_primitive_count: 0,
            transparent_or_degenerate_primitive_count: 0,
            timings: RtRepresentationTimings::default(),
            memory: RtRepresentationMemory::default(),
        }
    }

    pub fn format_log_line(self) -> String {
        let mut output = format!(
            "rt_representation_metrics schema={} representation={} pages={} exposed_faces={} candidate_primitives={} transparent_or_degenerate_primitives={}",
            RT_REPRESENTATION_METRICS_SCHEMA_VERSION,
            self.representation.label(),
            self.page_count,
            self.exposed_face_count,
            self.candidate_primitive_count,
            self.transparent_or_degenerate_primitive_count,
        );
        for (name, value) in [
            ("surface_generation_ms", self.timings.surface_generation_ms),
            ("omm_build_ms", self.timings.omm_build_ms),
            ("blas_build_ms", self.timings.blas_build_ms),
            ("blas_update_ms", self.timings.blas_update_ms),
            ("tlas_ms", self.timings.tlas_ms),
            ("primary_trace_ms", self.timings.primary_trace_ms),
            ("shadow_trace_ms", self.timings.shadow_trace_ms),
            ("gi_trace_ms", self.timings.gi_trace_ms),
        ] {
            match value {
                Some(value) => write!(output, " {name}={value:.4}").unwrap(),
                None => write!(output, " {name}=na").unwrap(),
            }
        }
        write!(
            output,
            " persistent_bytes={} scratch_bytes={}",
            self.memory.persistent_bytes, self.memory.scratch_bytes
        )
        .unwrap();
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rt_representation_metrics_expose_all_bakeoff_dimensions() {
        let metrics = RtRepresentationMetrics {
            representation: RtRepresentationKind::CompactExact,
            page_count: 7,
            exposed_face_count: 1_024,
            candidate_primitive_count: 2_048,
            transparent_or_degenerate_primitive_count: 0,
            timings: RtRepresentationTimings {
                surface_generation_ms: Some(0.25),
                omm_build_ms: None,
                blas_build_ms: Some(0.5),
                blas_update_ms: None,
                tlas_ms: Some(0.1),
                primary_trace_ms: Some(1.0),
                shadow_trace_ms: Some(0.75),
                gi_trace_ms: Some(1.5),
            },
            memory: RtRepresentationMemory {
                persistent_bytes: 65_536,
                scratch_bytes: 16_384,
            },
        };

        let line = metrics.format_log_line();

        for field in [
            "schema=1",
            "representation=compact_exact",
            "pages=7",
            "exposed_faces=1024",
            "candidate_primitives=2048",
            "transparent_or_degenerate_primitives=0",
            "surface_generation_ms=0.2500",
            "omm_build_ms=na",
            "blas_build_ms=0.5000",
            "blas_update_ms=na",
            "tlas_ms=0.1000",
            "primary_trace_ms=1.0000",
            "shadow_trace_ms=0.7500",
            "gi_trace_ms=1.5000",
            "persistent_bytes=65536",
            "scratch_bytes=16384",
        ] {
            assert!(
                line.contains(field),
                "missing stable metric field {field}: {line}"
            );
        }
    }

    #[test]
    fn rt_representation_metrics_keep_absent_stages_explicit() {
        let metrics = RtRepresentationMetrics::empty(RtRepresentationKind::ReferenceDda);

        assert_eq!(
            metrics.timings,
            RtRepresentationTimings {
                surface_generation_ms: None,
                omm_build_ms: None,
                blas_build_ms: None,
                blas_update_ms: None,
                tlas_ms: None,
                primary_trace_ms: None,
                shadow_trace_ms: None,
                gi_trace_ms: None,
            }
        );
        assert!(
            metrics
                .format_log_line()
                .contains("surface_generation_ms=na")
        );
        assert!(
            metrics
                .format_log_line()
                .contains("representation=reference_dda")
        );
    }

    #[test]
    fn rt_representation_kind_labels_are_stable_and_complete() {
        assert_eq!(
            RtRepresentationKind::ALL.map(RtRepresentationKind::label),
            [
                "reference_dda",
                "compact_exact",
                "hot_omm",
                "compact_greedy",
                "hot_interface",
                "mega_geometry",
            ]
        );
    }
}
