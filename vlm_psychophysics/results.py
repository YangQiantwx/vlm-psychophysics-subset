"""Exact-label scorer retained unchanged from the parent evaluation."""


def score(raw_answer, choices, truth, eligible, state, technical_pass):
    """Use exact labels after trimming only the returned answer's whitespace."""
    stripped = raw_answer.strip() if isinstance(raw_answer, str) else None
    prediction = stripped if stripped in choices else None
    if state == "PENDING":
        status = "MISSING"
    elif state == "INTENT":
        status = "UNCERTAIN"
    elif state != "RETURNED":
        raise ValueError(f"Unknown execution state {state!r}")
    elif technical_pass is not True:
        status = "TECHNICAL_FAILURE"
    elif prediction is None:
        status = "INVALID_ANSWER"
    else:
        status = "SCORED" if eligible else "UNSCORED"
    correct = (status == "SCORED" and prediction == truth) if eligible else None
    return prediction, status, correct
