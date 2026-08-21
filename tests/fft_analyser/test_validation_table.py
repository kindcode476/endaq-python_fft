"""Every row of the published validation table must hold when recomputed."""

import pathlib

import pytest

from fft_analyser.validation import validation_rows, render_markdown


ROWS = validation_rows()


@pytest.mark.parametrize("row", ROWS, ids=[r.test for r in ROWS])
def test_row_error_is_within_its_bound(row):
    assert row.max_error <= row.tolerance, (
        f"{row.test}: theory={row.theory!r} ours={row.ours!r} "
        f"ref={row.reference!r} err={row.max_error:.2e} "
        f"> tol={row.tolerance:.0e}")


def test_committed_table_matches_the_code():
    committed = (pathlib.Path(__file__).parents[2] / "fft_analyser"
                 / "VALIDATION.md")
    assert committed.exists(), \
        "regenerate with: python -m fft_analyser.validation"
    assert committed.read_text() == render_markdown(), \
        "VALIDATION.md is stale - regenerate with: python -m fft_analyser.validation"
