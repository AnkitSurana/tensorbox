"""
Tests for Continuous Training Pipeline.
"""

from pipeline import TrainingPipeline

def test_pipeline_promotion():
    dag = TrainingPipeline(baseline_f1=0.80)
    result = dag.run_cycle(records=500)
    assert result["status"] == "SUCCESS"
    assert result["promoted_to_prod"] is True
