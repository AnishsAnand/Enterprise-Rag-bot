User Selection Matching Prompt
You are an intelligent matching system. Match the user's input to the best option from the list.

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
{"matched": true, "matched_item": {"id": <exact_id_from_list>, "name": "<full_name_from_list>"}}

OR if truly no match:
{"matched": false}
Location Extraction Prompt
User query: "{user_query}"

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