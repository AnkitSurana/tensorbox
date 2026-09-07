"""
Tests for LoRA Serving Hub.
"""

from lora_server import LoRAServingHub

def test_lora_generation():
    hub = LoRAServingHub()
    res = hub.generate("Summarize liability clause", adapter_name="legal_v1")
    assert res["adapter_used"] == "legal_v1"
    assert "liability" in res["prompt"].lower()
