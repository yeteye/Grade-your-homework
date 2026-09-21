"""One-click aggregate entry for the 20 Module 2 AI-assisted test cases."""
import sys
from pathlib import Path

PARTS = Path(__file__).with_name("module2_parts")
sys.path.insert(0, str(PARTS))

from analytics_exports import AnalyticsExportTests  # noqa: E402,F401
from record_lifecycle import RecordLifecycleTests  # noqa: E402,F401
from security_protocol import SecurityProtocolTests  # noqa: E402,F401
from service_contract import ServiceContractTests  # noqa: E402,F401


if __name__ == "__main__":
    import unittest
    unittest.main(verbosity=2)
