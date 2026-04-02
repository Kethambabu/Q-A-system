# memory/tree_store.py
"""
Tree Store — Replaces the FAISS vector store.
Loads/saves hierarchical JSON trees and provides traversal + lookup utilities.
"""
import json
import os
from typing import Optional


TREE_DIR = os.path.join(os.path.dirname(__file__), "trees")


class TreeStore:
    """Manages persistence and traversal of document tree indices."""

    def __init__(self, storage_dir: str = TREE_DIR):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.tree: Optional[dict] = None
        self._node_index: dict = {}  # node_id -> node reference (flat lookup)

    # ───────────────────── Persistence ─────────────────────

    def save(self, tree: dict, filename: str = "tree_index.json") -> str:
        """Save tree to JSON file. Returns the file path."""
        filepath = os.path.join(self.storage_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(tree, f, indent=2, ensure_ascii=False)
        self.tree = tree
        self._build_index(tree)
        return filepath

    def load(self, filename: str = "tree_index.json") -> dict:
        """Load tree from JSON file."""
        filepath = os.path.join(self.storage_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Tree file not found: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            self.tree = json.load(f)
        self._build_index(self.tree)
        return self.tree

    def set_tree(self, tree: dict):
        """Set tree in memory without saving to disk."""
        self.tree = tree
        self._build_index(tree)

    # ───────────────────── Index / Lookup ─────────────────────

    def _build_index(self, node: dict, clear: bool = True):
        """Build flat index: node_id -> node reference for O(1) lookup."""
        if clear:
            self._node_index.clear()
        node_id = node.get("node_id")
        if node_id:
            self._node_index[node_id] = node
        for child in node.get("children", []):
            self._build_index(child, clear=False)

    def get_node(self, node_id: str) -> Optional[dict]:
        """O(1) lookup of a node by its ID."""
        return self._node_index.get(node_id)

    def get_nodes(self, node_ids: list) -> list:
        """Retrieve multiple nodes by their IDs."""
        return [self._node_index[nid] for nid in node_ids if nid in self._node_index]

    # ───────────────────── Traversal Helpers ─────────────────────

    def get_root(self) -> Optional[dict]:
        """Return the root node of the loaded tree."""
        return self.tree

    def get_children(self, node_id: str) -> list:
        """Return the children of a given node."""
        node = self.get_node(node_id)
        if node is None:
            return []
        return node.get("children", [])

    def get_children_summaries(self, node_id: str) -> list:
        """
        Return a list of {node_id, title, summary} for each child.
        This is what the traversal agent uses to decide which branch to follow.
        """
        children = self.get_children(node_id)
        return [
            {
                "node_id": c.get("node_id", ""),
                "title": c.get("title", ""),
                "summary": c.get("summary", ""),
            }
            for c in children
        ]

    def get_node_content(self, node_id: str) -> str:
        """Get the full content of a node (for context assembly)."""
        node = self.get_node(node_id)
        if node is None:
            return ""
        return node.get("content", "")

    def get_all_leaf_content(self, node_id: str) -> str:
        """Recursively collect all leaf content under a node."""
        node = self.get_node(node_id)
        if node is None:
            return ""
        children = node.get("children", [])
        if not children:
            return node.get("content", "")
        parts = []
        for child in children:
            parts.append(self.get_all_leaf_content(child.get("node_id", "")))
        return "\n\n".join(parts)

    def get_tree_depth(self, node: Optional[dict] = None) -> int:
        """Calculate the depth of the tree."""
        if node is None:
            node = self.tree
        if node is None:
            return 0
        children = node.get("children", [])
        if not children:
            return 1
        return 1 + max(self.get_tree_depth(c) for c in children)

    def get_node_count(self) -> int:
        """Total number of nodes in the tree."""
        return len(self._node_index)

    def list_tree_files(self) -> list:
        """List all saved tree JSON files."""
        if not os.path.exists(self.storage_dir):
            return []
        return [f for f in os.listdir(self.storage_dir) if f.endswith(".json")]
