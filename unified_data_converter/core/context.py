import os
import glob
import re
from typing import List, Dict, Optional, Any

class ContextNode:
    """
    Represents a logical node in the source data hierarchy (e.g., a specific Task or Episode).
    """
    def __init__(self, 
                 name: str, 
                 path: str, 
                 iterator_name: str, 
                 context_data: Dict[str, Any], 
                 parent: Optional['ContextNode'] = None):
        self.name = name                # e.g., "episode_0"
        self.path = path                # Absolute path to this unit
        self.iterator_name = iterator_name # e.g., "episode" (from hierarchy config)
        self.context_data = context_data # e.g., {"raw_id": "0", "task_name": "pick"}
        self.parent = parent
        self.children: List['ContextNode'] = []
    
    def add_child(self, child_node: 'ContextNode'):
        self.children.append(child_node)

    def get_full_context(self) -> Dict[str, Any]:
        """Merges context from root down to this node."""
        ctx = {}
        if self.parent:
            ctx.update(self.parent.get_full_context())
        ctx.update(self.context_data)
        return ctx

    def find_descendants(self, iterator_name: str) -> List['ContextNode']:
        """Recursively finds all descendants matching the given iterator name."""
        matches = []
        for child in self.children:
            if child.iterator_name == iterator_name:
                matches.append(child)
            # Continue searching down even if matched (though usually iterators are distinct levels)
            # Depending on structure, one might want to stop or continue. 
            # Assuming strictly hierarchical distinct levels, we can search deeper.
            matches.extend(child.find_descendants(iterator_name))
        return matches

    def __repr__(self):
        return f"<ContextNode {self.iterator_name}:{self.name} path={self.path}>"


class ContextScanner:
    """
    Scans the source directory based on hierarchy configuration to build a Context Tree.
    """
    def __init__(self, source_root: str):
        self.source_root = source_root

    def scan(self, hierarchy_config: List[Dict]) -> ContextNode:
        """
        Builds the context tree.
        Returns a root node representing the dataset level.
        """
        root_node = ContextNode(
            name="root",
            path=self.source_root,
            iterator_name="dataset",
            context_data={"dataset_root": self.source_root},
            parent=None
        )

        if not hierarchy_config:
            return root_node

        self._scan_recursive(root_node, hierarchy_config, 0)
        return root_node

    def _scan_recursive(self, parent_node: ContextNode, hierarchy_config: List[Dict], level_idx: int):
        if level_idx >= len(hierarchy_config):
            return

        level_config = hierarchy_config[level_idx]
        iterator_name = level_config["name"]
        node_type = level_config.get("type", "directory")
        pattern = level_config.get("pattern", "*")
        context_key = level_config.get("context_key", f"{iterator_name}_name")
        
        # Determine search path
        search_root = parent_node.path
        
        discovered_nodes = []

        if node_type == "directory":
            search_pattern = os.path.join(search_root, pattern)
            paths = glob.glob(search_pattern)
            
            for p in paths:
                if not os.path.isdir(p):
                    continue
                
                name = os.path.basename(p)
                
                # Regex extraction for name if specified
                # (Optional feature not in minimal spec but useful)
                
                node = ContextNode(
                    name=name,
                    path=p,
                    iterator_name=iterator_name,
                    context_data={context_key: name},
                    parent=parent_node
                )
                discovered_nodes.append(node)

        elif node_type == "filename_match":
            primary = level_config.get("primary_source", {})
            rel_pattern = primary.get("path", pattern) # Default to pattern if no primary.path
            id_regex = primary.get("id_regex", "(.*)")
            
            search_pattern = os.path.join(search_root, rel_pattern)
            files = glob.glob(search_pattern)
            
            # Map found IDs to their files to avoid duplicates if multiple files match (unlikely if regex captures ID)
            # Actually, we create one node per unique match
            
            seen_ids = set()
            
            for p in files:
                if not os.path.isfile(p):
                    continue
                
                filename = os.path.basename(p)
                match = re.search(id_regex, filename)
                
                if match:
                    # If regex has groups, use group 1, else whole match
                    extracted_id = match.group(1) if match.groups() else match.group(0)
                    
                    if extracted_id in seen_ids:
                        continue
                    seen_ids.add(extracted_id)
                    
                    # For filename match, the "path" of the node is usually the parent dir 
                    # because the node represents a logical unit within that dir, 
                    # but context has the ID.
                    # Or we can store the file path itself if it's the primary source.
                    # Let's store the directory as the path for consistency with navigation,
                    # but context will hold the ID.
                    
                    node = ContextNode(
                        name=extracted_id,
                        path=search_root, # It stays in the same dir
                        iterator_name=iterator_name,
                        context_data={context_key: extracted_id},
                        parent=parent_node
                    )
                    discovered_nodes.append(node)

        # Link and Recurse
        for node in discovered_nodes:
            parent_node.add_child(node)
            self._scan_recursive(node, hierarchy_config, level_idx + 1)

