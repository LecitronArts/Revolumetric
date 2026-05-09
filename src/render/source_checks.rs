pub(crate) fn normalize(source: &str) -> String {
    source.replace("\r\n", "\n").replace('\r', "\n")
}

pub(crate) fn read_source(path: &str) -> String {
    let source = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("failed to read source {path}: {error}"));
    normalize(&source)
}

pub(crate) fn compact(source: &str) -> String {
    source.split_whitespace().collect::<String>()
}

pub(crate) fn assert_contains_all(source: &str, tokens: &[&str], context: &str) {
    for token in tokens {
        assert!(source.contains(token), "{context} missing token {token}");
    }
}

pub(crate) fn assert_compact_contains_all(source: &str, tokens: &[&str], context: &str) {
    let compact_source = compact(source);
    for token in tokens {
        assert!(
            compact_source.contains(token),
            "{context} missing compact token {token}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_ignores_line_endings_and_spacing() {
        let source = "builder\r\n    .add_binding(\r\n        6,\r\n    )";
        assert_eq!(compact(source), "builder.add_binding(6,)");
        assert_compact_contains_all(source, &["builder.add_binding(6,)"], "compact source");
    }

    #[test]
    fn normalize_converts_crlf_and_lone_cr_to_lf() {
        assert_eq!(normalize("a\r\nb\rc"), "a\nb\nc");
    }
}
