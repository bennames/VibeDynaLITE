import sys
import pytest
from pathlib import Path

def test_profile():
    pytest.skip("Skip Metal GPU profiling test in sandboxed/headless environment")

