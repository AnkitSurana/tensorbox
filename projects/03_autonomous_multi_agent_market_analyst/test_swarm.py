"""
Tests for Multi-Agent Swarm.
"""

from swarm import SwarmOrchestrator

def test_swarm_execution():
    orchestrator = SwarmOrchestrator()
    res = orchestrator.analyze_equity("NVDA")
    assert res["ticker"] == "NVDA"
    assert len(res["agent_audit_trail"]) == 4
    assert res["verdict"] in ["BUY", "STRONG_BUY", "HOLD", "SELL"]
