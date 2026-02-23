"""Test ABCD requires 80% CD completion, not 50%."""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from candle_patterns import ABCD


class TestABCDCompletion:
    def test_config_cd_min_completion_default(self):
        """Default cd_min_completion should be 0.80."""
        detector = ABCD()
        assert detector.config["cd_min_completion"] == 0.80

    def test_config_cd_min_completion_override(self):
        """cd_min_completion should be configurable."""
        detector = ABCD(config={"cd_min_completion": 0.9})
        assert detector.config["cd_min_completion"] == 0.9
