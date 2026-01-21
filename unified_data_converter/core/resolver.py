import os
import re
from typing import Dict, Any, List, Union
from .context import ContextNode

class ParamResolver:
    def __init__(self, source_root: str, dest_root: str):
        self.source_root = source_root
        self.dest_root = dest_root

    def resolve(self, params: Dict[str, Any], context: Dict[str, Any], current_node: ContextNode) -> Dict[str, Any]:
        """
        Resolves a dictionary of parameters.
        """
        resolved = {}
        for k, v in params.items():
            resolved[k] = self._resolve_value(v, context, current_node)
        return resolved

    def _resolve_value(self, value: Any, context: Dict[str, Any], current_node: ContextNode) -> Any:
        if isinstance(value, str):
            return self._resolve_string(value, context)
        elif isinstance(value, list):
            return [self._resolve_value(item, context, current_node) for item in value]
        elif isinstance(value, dict):
            # Check if it is a Query Object
            if "select_iterator" in value:
                return self._resolve_aggregation(value, current_node)
            else:
                return {k: self._resolve_value(v, context, current_node) for k, v in value.items()}
        return value

    def _resolve_string(self, value: str, context: Dict[str, Any]) -> str:
        # 1. Variable Substitution
        # Use regex to replace {var} to handle missing keys gracefully or error
        def replace(match):
            key = match.group(1)
            if key in context:
                return str(context[key])
            return match.group(0) # Keep as is if not found
        
        resolved_str = re.sub(r'\{(\w+)\}', replace, value)

        # 2. Protocol Resolution
        if resolved_str.startswith("src://"):
            rel_path = resolved_str[6:]
            return os.path.join(self.source_root, rel_path)
        elif resolved_str.startswith("dest://"):
            rel_path = resolved_str[7:]
            return os.path.join(self.dest_root, rel_path)
        
        return resolved_str

    def _resolve_aggregation(self, query: Dict[str, Any], current_node: ContextNode) -> List[str]:
        iterator_name = query.get("select_iterator")
        target_file_tmpl = query.get("target_file", "")
        source_type = query.get("from", "source") # source or destination

        # 1. Find descendants
        descendants = current_node.find_descendants(iterator_name)
        
        results = []
        for node in descendants:
            node_context = node.get_full_context()
            
            # Format the target file string with the node's context
            # e.g. "video/{raw_id}.mp4" -> "video/0.mp4"
            # Note: We use the _resolve_string logic but with the node's context
            # We treat the template as a plain string first, then resolve protocols if present (unlikely for relative)
            
            # Simple substitution
            def replace(match):
                key = match.group(1)
                return str(node_context.get(key, match.group(0)))
            
            formatted_file = re.sub(r'\{(\w+)\}', replace, target_file_tmpl)
            
            if source_type == "source":
                # For source, we usually assume relative to the node's path if it's not a protocol
                if "://" in formatted_file:
                    # If user provided src:// in the template, respect it
                    path = self._resolve_string(formatted_file, node_context)
                else:
                    path = os.path.join(node.path, formatted_file)
            else: # destination
                # For destination, it's relative to dest_root (usually)
                # or if it has dest://
                if "://" in formatted_file:
                    path = self._resolve_string(formatted_file, node_context)
                else:
                    path = os.path.join(self.dest_root, formatted_file)
            
            results.append(path)
            
        return results

