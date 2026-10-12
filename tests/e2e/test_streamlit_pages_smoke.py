from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[2]


def _run_page(relative_path: str) -> AppTest:
    app = AppTest.from_file(str(ROOT / relative_path))
    app.run(timeout=20)
    return app


@pytest.mark.fast
@pytest.mark.e2e
def test_app_entrypoint_smoke_loads() -> None:
    app = _run_page("app.py")
    assert not app.exception


@pytest.mark.fast
@pytest.mark.e2e
@pytest.mark.parametrize(
    "page_path",
    [
        "pages/1_Lifecycle_Wealth_Path.py",
        "pages/2_Accumulation_Strategy_Design.py",
        "pages/2_DeRisking_Mechanics.py",
        "pages/3_Retirement_Sustainability.py",
        "pages/4_Sensitivity_Model_Risk.py",
    ],
)
def test_page_smoke_loads_without_exception(page_path: str) -> None:
    app = _run_page(page_path)
    assert not app.exception
