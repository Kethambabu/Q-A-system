"""Quick validation of all fixes."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Test 1: Query Refiner
print("=" * 50)
print("TEST 1: Query Refiner Agent")
print("=" * 50)
from agents.query_refiner_agent import refine_query

r1 = refine_query("give me numerical results")
print("Original:", r1["original_query"])
print("Refined: ", r1["refined_query"][:100])
print("Method:  ", r1["refinement_method"])
assert r1["refinement_method"] == "fast_path", "Expected fast_path for common pattern"
assert "accuracy" in r1["refined_query"].lower() or "metric" in r1["refined_query"].lower()
print("PASS\n")

r2 = refine_query("show me the results of the experiment")
print("Original:", r2["original_query"])
print("Refined: ", r2["refined_query"][:100])
print("Method:  ", r2["refinement_method"])
print("PASS\n")

# Test 2: Table Detection
print("=" * 50)
print("TEST 2: Table Detection & Structure Preservation")
print("=" * 50)
from utils.ingestion import _is_table_like, _contains_numerical_data, clean_text

table_text = "Model    Accuracy   F1-Score\nBERT     94.5%      92.3%\nGPT-2    91.2%      89.7%\nRoBERTa  95.1%      93.8%"
print("is_table_like:", _is_table_like(table_text))
print("has_numerical:", _contains_numerical_data(table_text))
assert _is_table_like(table_text), "Should detect as table"
assert _contains_numerical_data(table_text), "Should detect numbers"

preserved = clean_text(table_text, preserve_structure=True)
collapsed = clean_text(table_text, preserve_structure=False)
print("Preserved lines:", len(preserved.splitlines()))
print("Collapsed (single line):", "\n" not in collapsed)
assert len(preserved.splitlines()) >= 3, "Should preserve line structure"
assert "94.5%" in preserved, "Should preserve numbers"
print("PASS\n")

# Test 3: Numerical marker detection
print("=" * 50)
print("TEST 3: Numerical Marker Detection in Tree Builder")
print("=" * 50)
from agents.tree_builder_agent import _segment_has_numerical_markers

assert _segment_has_numerical_markers("[TABLE_DATA]\ntest\n[/TABLE_DATA]")
assert _segment_has_numerical_markers("accuracy was 94.5% and F1 was 92.3%")
assert not _segment_has_numerical_markers("This is a plain text paragraph about methods.")
print("PASS\n")

# Test 4: Numerical query detection
print("=" * 50)
print("TEST 4: Numerical Query Detection in Traversal")
print("=" * 50)
from agents.tree_traversal_agent import _is_numerical_query

assert _is_numerical_query("what are the numerical results?")
assert _is_numerical_query("show me accuracy and precision")
assert _is_numerical_query("performance metrics")
assert not _is_numerical_query("what is the introduction about?")
print("PASS\n")

# Test 5: Pipeline compiles
print("=" * 50)
print("TEST 5: LangGraph Pipeline Compilation")
print("=" * 50)
from core.pipeline import qa_graph
nodes = list(qa_graph.nodes.keys())
print("Pipeline nodes:", nodes)
assert "query_refiner_agent" in nodes, "Query refiner must be in pipeline"
assert "reasoning_agent" in nodes
assert "answer_agent" in nodes
print("PASS\n")

print("=" * 50)
print("ALL TESTS PASSED!")
print("=" * 50)
