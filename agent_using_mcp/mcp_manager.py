import os
import re
import json
import requests
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MCPResponse:
    success: bool
    data: Any
    error: Optional[str] = None

class MCPManager:
    """Handles all MCP-related functionality"""
    
    def __init__(self):
        self.mcps = {
            'joke': JokeMCP(),
            'eval': EvalMCP()
        }
        self.active_mcps = set(self.mcps.keys())
        
    def parse_query(self, query: str) -> tuple[str, str, List[str]]:
        """
        Parse query to determine which MCPs to use
        Returns: (processed_query, mcp_mode, selected_mcps)
        """
        mcp_pattern = r'@(\w+)'
        matches = re.findall(mcp_pattern, query)
        
        if matches:
            selected_mcps = [mcp for mcp in matches if mcp in self.mcps]
            processed_query = re.sub(mcp_pattern, '', query).strip()
            return processed_query, 'specific', selected_mcps
        
        keywords = {
            'joke': ['joke', 'funny', 'humor', 'dad joke', 'comedy'],
            'eval': ['calculate', 'math', 'compute', 'evaluate', 'expression', 'formula']
        }
        
        query_lower = query.lower()
        detected_mcps = []
        
        for mcp_name, mcp_keywords in keywords.items():
            if any(keyword in query_lower for keyword in mcp_keywords):
                detected_mcps.append(mcp_name)
        
        if detected_mcps:
            return query, 'keyword', detected_mcps
        
        return query, 'all', list(self.active_mcps)
    
    def execute_query(self, query: str) -> MCPResponse:
        """Execute query using appropriate MCPs"""
        try:
            processed_query, mode, selected_mcps = self.parse_query(query)
            
            if not selected_mcps:
                return MCPResponse(False, None, "No suitable MCP found for this query")
            
            if len(selected_mcps) == 1:
                mcp_name = selected_mcps[0]
                if mcp_name in self.mcps:
                    return self.mcps[mcp_name].execute(processed_query)
            
            results = []
            for mcp_name in selected_mcps:
                if mcp_name in self.mcps:
                    result = self.mcps[mcp_name].execute(processed_query)
                    if result.success:
                        return result
                    results.append(f"{mcp_name}: {result.error}")
            
            return MCPResponse(False, None, f"All MCPs failed: {'; '.join(results)}")
            
        except Exception as e:
            logger.error(f"MCP execution error: {e}")
            return MCPResponse(False, None, f"MCP execution error: {str(e)}")
    
    def list_mcps(self) -> Dict[str, str]:
        """List available MCPs and their descriptions"""
        return {
            name: mcp.get_description() 
            for name, mcp in self.mcps.items()
        }
    
    def toggle_mcp(self, mcp_name: str) -> bool:
        """Toggle MCP on/off"""
        if mcp_name not in self.mcps:
            return False
        
        if mcp_name in self.active_mcps:
            self.active_mcps.remove(mcp_name)
        else:
            self.active_mcps.add(mcp_name)
        
        return True


class BaseMCP:
    """Base class for all MCPs"""
    
    def execute(self, query: str) -> MCPResponse:
        raise NotImplementedError
    
    def get_description(self) -> str:
        raise NotImplementedError


class JokeMCP(BaseMCP):
    """Dad Jokes MCP using icanhazdadjoke.com API"""
    
    def __init__(self):
        self.base_url = "https://icanhazdadjoke.com"
        self.headers = {
            "Accept": "application/json",
            "User-Agent": "Basic Agent MCP (github.com/user/basic-agent)"
        }
    
    def execute(self, query: str) -> MCPResponse:
        try:
            if "search" in query.lower():
                return self._search_jokes(query)
            elif "specific" in query.lower() or "id:" in query.lower():
                return self._get_specific_joke(query)
            else:
                return self._get_random_joke()
                
        except Exception as e:
            return MCPResponse(False, None, f"Joke API error: {str(e)}")
    
    def _get_random_joke(self) -> MCPResponse:
        try:
            response = requests.get(self.base_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            return MCPResponse(True, {
                "joke": data["joke"],
                "id": data["id"],
                "type": "random"
            })
        except Exception as e:
            return MCPResponse(False, None, f"Failed to get random joke: {str(e)}")
    
    def _search_jokes(self, query: str) -> MCPResponse:
        try:
            search_term = query.lower().replace("search", "").strip()
            if not search_term:
                search_term = "funny"
            
            params = {"term": search_term, "limit": 5}
            response = requests.get(f"{self.base_url}/search", headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data["results"]:
                return MCPResponse(True, {
                    "jokes": data["results"],
                    "total": data["total_jokes"],
                    "search_term": search_term,
                    "type": "search"
                })
            else:
                return MCPResponse(False, None, f"No jokes found for '{search_term}'")
                
        except Exception as e:
            return MCPResponse(False, None, f"Failed to search jokes: {str(e)}")
    
    def _get_specific_joke(self, query: str) -> MCPResponse:
        try:
            id_match = re.search(r'id:([A-Za-z0-9]+)', query)
            if not id_match:
                return MCPResponse(False, None, "No valid joke ID found in query")
            
            joke_id = id_match.group(1)
            response = requests.get(f"{self.base_url}/j/{joke_id}", headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return MCPResponse(True, {
                "joke": data["joke"],
                "id": data["id"],
                "type": "specific"
            })
            
        except Exception as e:
            return MCPResponse(False, None, f"Failed to get specific joke: {str(e)}")
    
    def get_description(self) -> str:
        return "Fetches dad jokes from icanhazdadjoke.com - supports random, search, and specific jokes"


class EvalMCP(BaseMCP):
    """Safe Python expression evaluator MCP"""
    
    def __init__(self):
        import math
        self.allowed_names = {
            k: v for k, v in math.__dict__.items() 
            if not k.startswith("__")
        }
        self.allowed_names.update({
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'sum': sum, 'len': len, 'int': int, 'float': float,
            'str': str, 'bool': bool, 'list': list, 'tuple': tuple,
            'dict': dict, 'set': set
        })
    
    def execute(self, query: str) -> MCPResponse:
        try:
            expression = self._extract_expression(query)
            if not expression:
                return MCPResponse(False, None, "No valid expression found in query")
            
            result = self._safe_eval(expression)
            return MCPResponse(True, {
                "expression": expression,
                "result": result,
                "type": "evaluation"
            })
            
        except Exception as e:
            return MCPResponse(False, None, f"Evaluation error: {str(e)}")
    
    def _extract_expression(self, query: str) -> str:
        """Extract mathematical expression from natural language query"""
        prefixes = [
            "calculate", "compute", "evaluate", "what is", "what's",
            "solve", "find", "determine", "math", "expression"
        ]
        
        expression = query.lower()
        for prefix in prefixes:
            expression = expression.replace(prefix, "").strip()
        
        expression = re.sub(r'[?!]', '', expression)
        
        expression = expression.replace("plus", "+")
        expression = expression.replace("minus", "-")
        expression = expression.replace("times", "*")
        expression = expression.replace("divided by", "/")
        expression = expression.replace("to the power of", "**")
        expression = expression.replace("squared", "**2")
        expression = expression.replace("cubed", "**3")
        
        return expression.strip()
    
    def _safe_eval(self, expression: str) -> Any:
        """Safely evaluate mathematical expression"""
        try:
            code = compile(expression, "<string>", "eval")
            
            for name in code.co_names:
                if name not in self.allowed_names:
                    raise NameError(f"Use of '{name}' is not allowed")
            
            result = eval(code, {"__builtins__": {}}, self.allowed_names)
            return result
            
        except Exception as e:
            raise Exception(f"Expression evaluation failed: {str(e)}")
    
    def get_description(self) -> str:
        return "Safely evaluates mathematical expressions with access to math functions and constants"
