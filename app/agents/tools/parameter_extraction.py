"""
Parameter Extraction Tools - Reusable utilities for extracting and matching parameters.
"""

import logging
import json
from typing import List, Dict, Any, Optional

from app.services.ai_service import ai_service

logger = logging.getLogger(__name__)


class ParameterExtractor:
    """
    Tools for extracting and matching parameters from user input using LLM.
    """
    
    async def match_user_selection(
        self,
        user_input: str,
        available_options: List[Dict[str, Any]]
    ) -> str:
        """
        Use LLM to intelligently match user's input to available options.
        
        Handles:
        - Exact matches
        - Partial matches  
        - Aliases and typos
        - "all" selection
        - Multiple selections (comma-separated)
        
        Args:
            user_input: User's response
            available_options: List of dicts with 'id' and 'name' keys
            
        Returns:
            JSON string with match result
        """
        try:
            logger.info(f"🔍 Matching user input: '{user_input}' against {len(available_options)} options")
            
            # Format options for LLM
            formatted_options = "\n".join([
                f"- {opt.get('name', opt.get('itemName', str(opt)))}" 
                for opt in available_options
            ])
            
            prompt = f"""The user was shown these options:
{formatted_options}

The user responded with: "{user_input}"

Match the user's response to the available options. Handle:
- Exact matches (case-insensitive)
- Partial matches (e.g., "delhi" matches "Delhi")
- Multiple selections (e.g., "delhi, bengaluru" or "delhi and mumbai")
- "all" means select all options
- Typos and aliases

Respond with ONLY valid JSON in this format:
{{"matched": true, "matched_item": {{"id": ..., "name": "..."}}, "matched_ids": [...], "matched_names": [...]}}

OR if no match:
{{"matched": false, "no_match": true}}

If "all" was requested:
{{"matched": true, "all": true, "matched_ids": [...all ids...], "matched_names": [...all names...]}}

If multiple matches:
{{"matched": true, "multiple": true, "matched_ids": [...], "matched_names": [...]}}
"""
            
            llm_response = await ai_service._call_chat_with_retries(
                prompt=prompt,
                temperature=0.0,
                max_tokens=500
            )
            
            # Clean response (remove markdown if present)
            response_text = llm_response.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("\n", 1)[1]
                if response_text.endswith("```"):
                    response_text = response_text.rsplit("\n", 1)[0]
            
            # Parse JSON
            result = json.loads(response_text)
            
            if result.get("matched"):
                if result.get("all"):
                    logger.info(f"✅ Matched ALL options")
                    return json.dumps({
                        "matched": True,
                        "all": True,
                        "matched_ids": result.get("matched_ids", []),
                        "matched_names": result.get("matched_names", [])
                    }, indent=2)
                elif result.get("multiple"):
                    logger.info(f"✅ Matched MULTIPLE: {result.get('matched_names')}")
                    return json.dumps({
                        "matched": True,
                        "multiple": True,
                        "matched_ids": result.get("matched_ids", []),
                        "matched_names": result.get("matched_names", [])
                    }, indent=2)
                else:
                    matched_item = result.get("matched_item", {})
                    logger.info(f"✅ Matched: {matched_item.get('name')}")
                    return json.dumps({
                        "matched": True,
                        "matched_item": matched_item
                    }, indent=2)
            
            # No match
            logger.info("❌ No match found")
            return json.dumps({"matched": False, "no_match": True}, indent=2)
            
        except Exception as e:
            logger.error(f"Error matching selection: {e}")
            return json.dumps({"matched": False, "error": str(e)}, indent=2)
    
    async def extract_location_from_query(
        self,
        user_query: str,
        available_endpoints: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Extract location/endpoint names from user query using LLM.
        
        Args:
            user_query: User's query
            available_endpoints: List of available endpoints
            
        Returns:
            Extracted location string or None
        """
        try:
            endpoint_names = [ep.get("endpointDisplayName", ep.get("name", "")) for ep in available_endpoints]
            endpoints_str = ", ".join(endpoint_names)
            
            prompt = f"""User query: "{user_query}"

Available locations: {endpoints_str}

Extract location names from the query. Handle:
- Single location: "clusters in Delhi" → Delhi
- Multiple locations: "clusters in delhi and bengaluru" → Delhi, Bengaluru
- "all" or "all clusters" → LOCATION: all
- No location mentioned → LOCATION: none

Respond with ONLY:
- LOCATION: <name(s)>
- LOCATION: all
- LOCATION: none

Examples:
"list clusters in Mumbai" → LOCATION: Mumbai
"show clusters in delhi and bengaluru" → LOCATION: Delhi, Bengaluru
"list all clusters" → LOCATION: all
"count clusters" → LOCATION: all
"what are the clusters?" → LOCATION: all
"""
            
            llm_response = await ai_service._call_chat_with_retries(
                prompt=prompt,
                temperature=0.0,
                max_tokens=100
            )
            
            result = llm_response.strip()
            
            if result.startswith("LOCATION:"):
                location = result.replace("LOCATION:", "").strip()
                logger.info(f"🌍 Extracted location: {location}")
                return location
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting location: {e}")
            return None

