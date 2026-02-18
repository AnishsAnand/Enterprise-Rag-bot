Query Intent Analysis Prompt
Analyze this search query:
"{query}"

Provide:
1. Query type (factual/procedural/conceptual/comparative)
2. Key entities (extract specific names, versions, technologies)
3. Implicit requirements (what user really needs)
4. Suggested expansions (related terms to search)

Respond in JSON:
{
  "type": "...",
  "entities": ["...", "..."],
  "implicit_needs": ["...", "..."],
  "expansions": ["...", "..."],
  "is_technical": true/false
}
Task Intent Detection Prompt
Analyze the following user query and determine the primary task intent.

User Query: {query}

Classify the intent into one of these categories:
- scrape: User wants to extract data from a website/URL
- search: User wants to find information from knowledge base
- analyze: User wants analysis, explanation, or summary
- upload: User wants to process/upload a file or document
- bulk_operation: User wants to scrape multiple pages/URLs
- unknown: Intent is unclear

Respond ONLY with valid JSON in this format:
{
  "type": "category_name",
  "confidence": 0.0-1.0,
  "extracted_params": ["param1", "param2"],
  "reasoning": "brief explanation"
}
Query Expansion Prompt
Given the search query: "{query}"

Generate 3-5 highly relevant related terms, synonyms, or alternative phrasings that would help find relevant documentation.

{f"Context: {', '.join(context_hints[:3])}" if context_hints else ""}

Respond with just the terms, comma-separated:
Response Generation Prompt (Informational)
Provide a comprehensive and accurate answer based on the context below. 
Focus on factual information and cite specific details from the sources. 
Structure your response logically with clear sections. 
If information is insufficient, clearly state what's missing.
Response Generation Prompt (Instructional)
Give clear, step-by-step instructions based on the information provided. 
Ensure accuracy and completeness. Number each step clearly. 
Include warnings or important notes where relevant. 
Verify all technical details against the provided context.
Response Generation Prompt (Troubleshooting)
Analyze the issue systematically and provide a structured solution with clear steps. 
Reference specific technical details from the context. 
Prioritize the most common solutions first. 
Include diagnostic steps if applicable.
Response Generation Prompt (Explanatory)
Explain the concept clearly with examples and practical applications. 
Use the provided context to ensure accuracy. 
Break down complex ideas into understandable parts. 
Include relevant analogies or comparisons when helpful.
Context Expansion Prompt
Analyze the following context and create a comprehensive, accurate summary that directly relates to the user's query.

User Query: {query}

Context to analyze:
{context_text}

Instructions:
1. Extract ALL key facts and information that directly answer or relate to the query
2. Organize information logically with clear structure
3. Maintain absolute accuracy - don't add information not present in the context
4. Be comprehensive but concise - include technical details and specific examples
5. Focus on actionable information when applicable
6. Preserve important numbers, dates, and technical specifications
7. If there are multiple perspectives or approaches, include them all

Expanded Context:
Enhanced Response Generation Prompt
{template}

IMPORTANT INSTRUCTIONS:
1. Base your answer STRICTLY on the provided context
2. Cite sources using [Source N] notation when referencing specific information
3. If information is insufficient, clearly state what's missing
4. Provide specific examples and details from the context
5. Organize your response with clear structure

User Query: {query}

Available Context:
{combined_context}

Your Response (with source citations):
Summary Generation Prompt
Create a concise, informative summary of the following content.

Requirements:
- Maximum {max_sentences} sentences
- Focus on key points and main ideas
- Use clear, professional language
- Avoid repetition and filler words
- Include specific details and facts when relevant
- Make it actionable and useful

Content:
{text}

Summary:
Stepwise Response Generation Prompt
Provide clear, actionable step-by-step instructions based on the context below.

Context:
{context}

Available Images (use ONLY if relevant):
{available_images_json}

User Question: {query}

INSTRUCTIONS:

1. Generate {max_steps} clear, specific steps in {style}

2. For images:
   - If a relevant image exists in Available Images, include it:
     "image": {"url": "exact_url_from_above", "alt": "description", "caption": "optional"}
   - If NO relevant image exists, OMIT the image field entirely
   - Do NOT create image_prompt or placeholder descriptions
   - Only include images that directly illustrate the step

3. Each step must be specific, clear, and actionable

4. Use appropriate step types:
   - "action": For steps requiring user action
   - "note": For warnings, tips, or cautions
   - "info": For context or background

5. Return ONLY valid JSON array (no markdown, no preamble):

[
  {
    "step_number": 1,
    "text": "Clear step description",
    "type": "action",
    "image": {"url": "https://...", "alt": "...", "caption": "..."}
  },
  {
    "step_number": 2,
    "text": "Another step description",
    "type": "action"
  }
]

IMPORTANT:
- Steps WITHOUT matching images should not have an "image" field at all
- Do NOT generate placeholder image descriptions
- Only include real image URLs from the Available Images list