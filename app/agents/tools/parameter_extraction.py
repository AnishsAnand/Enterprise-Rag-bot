"""
Parameter Extraction Tools - Reusable utilities for extracting and matching parameters.
"""

import logging
import json
import re
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
            
            # Format options for LLM - INCLUDE IDs so LLM knows the correct ID to return
            formatted_options = "\n".join([
                f"- ID: {opt.get('id')}, Name: {opt.get('name', opt.get('itemName', str(opt)))}" 
                for opt in available_options
            ])
            
            prompt = f"""You are an intelligent matching system. Match the user's input to the best option from the list.

Available Options:
{formatted_options}

User Input: "{user_input}"

CRITICAL MATCHING RULES:
1. **Abbreviations**: Match abbreviations intelligently
   - "Ind South" → "India South(Bengaluru)"
   - "Ind North" → "India North(Delhi)"
   - "Ind Central" → "India Central(GCCMumbai)" or "India Central(Mumbai-BK)"
   - "Delhi" → "India North(Delhi)" or "India North(GCCDelhi)"
   - "Bengaluru" or "Bangalore" or "BLR" → "India South(Bengaluru)"
   - "Mumbai" or "Mum" → "India Central(Mumbai-BK)" or "India Central(GCCMumbai)" or "India East(Mumbai-DC3)"
   - "Chennai" → "India South(Chennai-AMB)"

2. **Partial Word Matching**: Match any part of the option name
   - "South" matches any option containing "South"
   - "North" matches any option containing "North"
   - "Bengaluru" matches "India South(Bengaluru)"
   - "Delhi" matches "India North(Delhi)" or "India North(GCCDelhi)"

3. **Case-Insensitive**: "ind south" = "Ind South" = "IND SOUTH"

4. **Word Order**: "South India" should match "India South"

5. **Common Aliases**:
   - "BLR" = "Bengaluru" = "Bangalore"
   - "DEL" = "Delhi"
   - "MUM" = "Mumbai"
   - "CHN" = "Chennai"

6. **Multiple Selections**: Handle comma-separated or "and" separated inputs
   - "delhi, bengaluru" → match both
   - "delhi and mumbai" → match both

7. **"all"**: If user says "all", select all options

EXAMPLES:
- User: "Ind South" → Match: "India South(Bengaluru)" (ID from list)
- User: "Bengaluru" → Match: "India South(Bengaluru)" (ID from list)
- User: "South" → Match: "India South(Bengaluru)" or "India South(Chennai-AMB)" (pick most common)
- User: "delhi" → Match: "India North(Delhi)" (ID from list)
- User: "mumbai" → Match: "India Central(Mumbai-BK)" (ID from list)

YOU MUST:
- Always return a match if there's ANY reasonable connection
- Use the EXACT ID from the options list above
- Return the FULL option name (not abbreviated)
- If multiple options match, pick the most common/obvious one

Respond with ONLY valid JSON (no markdown, no explanation):
{{"matched": true, "matched_item": {{"id": <exact_id_from_list>, "name": "<full_name_from_list>"}}}}

OR if truly no match:
{{"matched": false}}
"""
            
            # Use slightly higher temperature for more creative/intelligent matching
            llm_response = await ai_service._call_chat_with_retries(
                prompt=prompt,
                temperature=0.2,  # Slightly higher for better abbreviation handling
                max_tokens=500
            )
            
            # Clean response (remove markdown, code blocks, explanations)
            response_text = llm_response.strip()
            
            # Remove markdown code blocks
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            elif response_text.startswith("```"):
                response_text = response_text.split("```")[1].strip()
                if response_text.startswith("json"):
                    response_text = response_text[4:].strip()
            
            # Extract JSON if there's extra text
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(0)
            
            # Parse JSON with retry logic
            result = None
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️ First JSON parse failed: {e}, trying to extract JSON...")
                # Try to find JSON object in the response
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', llm_response, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group(0))
                        logger.info(f"✅ Successfully extracted JSON from response")
                    except json.JSONDecodeError:
                        logger.error(f"❌ Failed to parse JSON even after extraction: {llm_response[:200]}")
                        return json.dumps({"matched": False, "error": "Invalid JSON response from LLM"}, indent=2)
                else:
                    logger.error(f"❌ No JSON found in response: {llm_response[:200]}")
                    return json.dumps({"matched": False, "error": "No JSON found in LLM response"}, indent=2)
            
            if result is None:
                logger.error(f"❌ Failed to parse JSON: {llm_response[:200]}")
                return json.dumps({"matched": False, "error": "Failed to parse LLM response"}, indent=2)
            
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
