"""
Small Language Model (SLM) component for generating natural language
investigation reports from structured anomaly detection outputs.
Uses quantized Phi-3 or Llama-3-8B for edge deployment.
"""
import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from llama_cpp import Llama

logger = logging.getLogger(__name__)


class TheftReportGenerator:
    """
    Generates natural language investigation reports using a quantized SLM.
    Optimized for utility company edge deployment with GGUF/AWQ quantization.
    """

    # Structured prompt template - no raw time-series textification
    SYSTEM_PROMPT = """You are an expert electricity fraud analyst for a utility company.
Analyze the provided structured anomaly detection data and generate a concise investigation report.
Focus on actionable insights for field inspection teams. Be specific about consumption patterns."""

    USER_PROMPT_TEMPLATE = """## Consumer Anomaly Detection Report

**Consumer ID**: {consumer_id}
**Risk Assessment**: {risk_tier} (Probability: {fraud_probability:.1%})
**Anomaly Score**: {anomaly_score:.3f}

### Key Anomalous Features Detected:
{key_features_formatted}

### Consumption Pattern Summary:
- Primary anomaly driver: {primary_feature}
- Pattern type: {pattern_type}

### Investigation Instructions:
Generate a 2-3 sentence investigation summary for the field team.
Identify the specific consumption behavior that triggered this alert.
Suggest inspection priority and focus areas (meter tampering, bypass, or data manipulation).
Do not list raw numbers; interpret the pattern in operational terms."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        backend: str = "llama.cpp",  # or "vllm", "transformers"
        max_tokens: int = 256,
        temperature: float = 0.3,
    ):
        """
        Initialize report generator with quantized SLM.

        Args:
            model_path: Path to GGUF/AWQ quantized model
            backend: Inference backend optimized for edge deployment
            max_tokens: Generation length limit
            temperature: Creativity vs determinism (low for factual reports)
        """
        self.model_path = model_path
        self.backend = backend
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model: Optional["Llama"] = None
        self.sampling_params: Any = None
        self.tokenizer = None

        if model_path:
            self._load_model()

    def _load_model(self):
        if not self.model_path:
            return

        if self.backend == "llama.cpp":
            from llama_cpp import Llama

            # Remove the ': Llama' here to prevent the [no-redef] error
            self.model = Llama(model_path=self.model_path, n_ctx=2048, n_threads=4, verbose=False)

    def generate_report(self, detection_result) -> str:
        """
        Generate natural language report from structured detection result.
        Avoids token-inflation by using structured data, not raw time-series text.
        """
        # Format key features for prompt
        key_features_str = "\n".join(
            [
                f"- {feat}: {val:+.3f} impact"
                for feat, val in list(detection_result.key_features.items())[:3]
            ]
        )

        # Determine pattern type from features
        pattern_type = self._classify_pattern(detection_result.key_features)

        # Get primary feature
        primary = max(detection_result.key_features.items(), key=lambda x: abs(x[1]))

        prompt = self.USER_PROMPT_TEMPLATE.format(
            consumer_id=detection_result.consumer_id,
            risk_tier=detection_result.risk_tier,
            fraud_probability=detection_result.fraud_probability,
            anomaly_score=detection_result.anomaly_score,
            key_features_formatted=key_features_str,
            primary_feature=primary[0],
            pattern_type=pattern_type,
        )

        if self.model is None:
            # Fallback template-based generation if no SLM loaded
            return self._template_report(detection_result, pattern_type)

        # Generate with SLM

        if self.backend == "llama.cpp":
            output = self.model.create_chat_completion(
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            # To:
            if isinstance(output, dict):
                report = output["choices"][0]["message"]["content"]
            else:
                # Handle streaming response (Iterator case)
                report = "".join(
                    [chunk["choices"][0]["delta"].get("content", "") for chunk in output]
                )

        elif self.backend == "vllm":
            # from vllm import SamplingParams

            prompt_tokens = self.model.tokenize(prompt.encode("utf-8"))
            outputs = self.model.generate(prompt_tokens, self.sampling_params)
            report = "".join(
                [
                    self.model.detokenize([token]).decode("utf-8", errors="ignore")
                    for token in outputs
                ]
            )

        else:
            report = ""

        # Clean up output
        report = self._post_process(report)
        return report

    def _classify_pattern(self, key_features: Dict[str, float]) -> str:
        """Classify theft pattern from feature importance."""
        features_str = " ".join(key_features.keys()).lower()

        if "zero" in features_str or "strike_below" in features_str:
            return "Zero Consumption / Meter Bypass"
        elif "trend" in features_str and any(
            v < 0 for k, v in key_features.items() if "trend" in k
        ):
            return "Declining Consumption Trend"
        elif "sudden_drop" in features_str:
            return "Sudden Usage Reduction"
        elif "standard_deviation" in features_str or "variance" in features_str:
            return "Abnormal Consumption Stability"
        else:
            return "General Anomaly Pattern"

    def _template_report(self, detection_result, pattern_type: str) -> str:
        """Fallback template-based report when SLM unavailable."""
        tier = detection_result.risk_tier
        prob = detection_result.fraud_probability

        templates = {
            "CRITICAL": (
                f"URGENT: Consumer {detection_result.consumer_id} "
                f"shows strong indicators of electricity theft. "
                f"Pattern suggests {pattern_type} with {prob:.1%} confidence. "
                f"Recommend immediate physical inspection of meter installation."
            ),
            "HIGH": (
                f"Consumer {detection_result.consumer_id} "
                f"exhibits suspicious consumption anomalies consistent with "
                f"{pattern_type}. Probability of fraud: {prob:.1%}. "
                f"Schedule inspection within 48 hours."
            ),
            "MEDIUM": (
                f"Anomalous patterns detected for {detection_result.consumer_id}. "
                f"Possible {pattern_type}. Monitor consumption over next billing cycle."
            ),
            "LOW": (
                f"Minor anomalies detected. No immediate action "
                f"required for {detection_result.consumer_id}."
            ),
        }
        return templates.get(tier, templates["MEDIUM"])

    def _post_process(self, text: str) -> str:
        """Clean and format generated text."""
        # Remove any markdown code blocks
        text = re.sub(r"```[\s\S]*?```", "", text)
        # Remove multiple newlines
        text = re.sub(r"\n+", "\n", text).strip()
        # Ensure single space after periods
        text = re.sub(r"\.(?!\s)", ". ", text)
        return text

    def batch_generate(self, detection_results: List) -> Dict[str, str]:
        """Generate reports for multiple consumers efficiently."""
        reports = {}
        for result in detection_results:
            reports[result.consumer_id] = self.generate_report(result)
        return reports

    # Updated logic in src/llm/report_generator.py
    def _format_context(self, detection_result):
        from src.schemas.feature_map import FEATURE_TRANSLATIONS

        translated_features = []
        for feat, impact in detection_result.key_features.items():
            readable_name = FEATURE_TRANSLATIONS.get(feat, feat)
            direction = "Positive" if impact > 0 else "Negative"
            translated_features.append(f"- {readable_name}: {direction} impact on risk")

        return "\n".join(translated_features)
