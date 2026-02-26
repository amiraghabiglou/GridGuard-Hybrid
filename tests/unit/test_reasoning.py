# tests/unit/test_reasoning.py
import pytest

from src.llm.report_generator import TheftReportGenerator
from src.models.ensemble import DetectionResult
from src.schemas.feature_map import FEATURE_TRANSLATIONS, THEFT_PERSONAS


@pytest.fixture
def mock_detection_result():
    return DetectionResult(
        consumer_id="test_123",
        anomaly_score=0.85,
        fraud_probability=0.92,
        risk_tier="CRITICAL",
        key_features={
            "domain__zero_consumption_ratio": 0.45,
            'value__linear_trend__attr_"slope"': -0.12,
        },
        explanation="High risk of bypass",
    )


def test_feature_translation_integrity():
    """Ensure all critical TSFRESH features have human-readable mappings."""
    required_keys = ['value__linear_trend__attr_"slope"', "domain__zero_consumption_ratio"]
    for key in required_keys:
        assert key in FEATURE_TRANSLATIONS
        assert isinstance(FEATURE_TRANSLATIONS[key], str)


def test_report_context_formatting(mock_detection_result):
    """Verify that technical keys are translated before reaching the SLM."""
    generator = TheftReportGenerator()
    context = generator._format_context(mock_detection_result)

    # Context should use human-readable names, not raw TSFRESH keys
    assert "Percentage of days with zero usage" in context
    assert "domain__zero_consumption_ratio" not in context


def test_persona_classification(mock_detection_result):
    """Verify pattern classification logic maps correctly to theft personas."""
    generator = TheftReportGenerator()
    pattern = generator._classify_pattern(mock_detection_result.key_features)

    assert pattern == "Zero Consumption / Meter Bypass"
