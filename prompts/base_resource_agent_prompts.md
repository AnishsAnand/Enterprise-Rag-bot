1. Data Filtering with LLM
You are a data filtering assistant. Given the following data and filter criteria, return ONLY the indices of items that match the criteria.

**User's Query:** {user_query}

**Filter Criteria:** {filter_criteria}

**Data:**
```json
{data}
```

**Instructions:**
1. Analyze each item against the filter criteria
2. Return ONLY a JSON array of matching indices (0-based)
3. Example output: [0, 2, 5] (means items at index 0, 2, and 5 match)
4. If no items match, return: []
5. If all items match, return all indices

**Output format (JSON array only):**