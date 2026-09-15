from open_endurance_coach.tokens import CHARS_PER_TOKEN, estimate_text_tokens


def test_estimate_is_within_calibrated_error_of_the_measured_prompt() -> None:
    measured_chars = 12380
    measured_tokens = 4039
    estimate = estimate_text_tokens("x" * measured_chars)
    assert measured_tokens <= estimate <= int(measured_tokens * 1.05)
    assert CHARS_PER_TOKEN == 3
