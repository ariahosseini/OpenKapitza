"""
Unit and regression test for the OpenKapitza package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import OpenKapitza


def test_OpenKapitza_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "OpenKapitza" in sys.modules
