# src/workers/tasks.py
from celery import Celery

from src.features.extractors import ElectricityFeatureExtractor
from src.llm.report_generator import TheftReportGenerator
from src.models.ensemble import HybridTheftDetector

celery_app = Celery("tasks", broker="redis://redis:6379/0", backend="redis://redis:6379/0")


@celery_app.task(bind=True, name="process_theft_analysis")
def process_theft_analysis(self, consumer_data_batch):
    """
    Background task to handle the CPU-intensive pipeline.
    """
    # 1. Feature Extraction (The Slowest Part)
    extractor = ElectricityFeatureExtractor(n_jobs=1)  # Reduce parallel overhead in worker
    df_long = extractor.prepare_data(pd.DataFrame(consumer_data_batch))
    features = extractor.extract_features(df_long)

    # 2. Prediction
    detector = HybridTheftDetector.load("models/hybrid_detector.joblib")
    # Only calculate SHAP if necessary
    results = detector.predict(features, calculate_shap=True)

    # 3. Reasoning (The Memory-Intensive Part)
    # Note: In a larger scale, this would be a separate microservice call
    report_gen = TheftReportGenerator(model_path="models/phi-3-q4.gguf")

    final_output = []
    for res in results:
        report = None
        if res.fraud_probability > 0.6:  # High-risk threshold
            report = report_gen.generate_report(res)

        final_output.append({**res.dict(), "report": report})

    return final_output
