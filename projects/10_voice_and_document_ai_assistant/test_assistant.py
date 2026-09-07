"""
Tests for Voice & Document AI Assistant.
"""

from assistant import VoiceDocAssistant

def test_voice_doc_assistant():
    bot = VoiceDocAssistant()
    bot.ingest_document("Tensorbox deployment instructions and user manual.")
    res = bot.process_voice_query("How do I deploy on port 5001?")
    assert "deployment" in res["grounding_doc"]
    assert "Synthesized" in res["spoken_response"]
