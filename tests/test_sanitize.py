from open_endurance_coach.sanitize import sanitize_text


def test_replaces_terminal_escapes_and_control_characters() -> None:
    assert sanitize_text("\x1b[2J\x1b[1A done") == "\ufffd[2J\ufffd[1A done"
    assert sanitize_text("a\x00b\x07c\x7fd") == "a\ufffdb\ufffdc\ufffdd"
    assert sanitize_text("csi \x9b1m end") == "csi \ufffd1m end"


def test_normalises_carriage_returns_to_newlines() -> None:
    assert sanitize_text("a\r\nb\rc") == "a\nb\nc"


def test_keeps_newlines_tabs_and_printable_unicode() -> None:
    text = "ligne 1\n\tindenté éàü 漢字 🚴"
    assert sanitize_text(text) == text


def test_replaces_bidi_overrides_and_line_separators() -> None:
    assert sanitize_text("a\u2028b\u2029c\u0085d") == "a\ufffdb\ufffdc\ufffdd"
    assert sanitize_text("x\u202ey\u2066z") == "x\ufffdy\ufffdz"


def test_keeps_weak_direction_marks_for_non_latin_text() -> None:
    text = "مرحبا\u061c\u200f שלום\u200e"
    assert sanitize_text(text) == text


def test_replaces_unpaired_surrogates_and_stays_encodable() -> None:
    cleaned = sanitize_text("ok \ud800 bad \udfff end")
    assert cleaned == "ok \ufffd bad \ufffd end"
    cleaned.encode("utf-8")
