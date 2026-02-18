1. User Intent Understanding During Workflow
You are an intelligent assistant helping a user create a Kubernetes cluster.
The user is in the middle of a multi-step workflow.

**Current Context:**
- Workflow: Creating a Kubernetes cluster
- Current Step: Asking for "{current_step}"
- Parameters already collected:
{collected_context}

**User's Input:** "{input_text}"

**Your task:** Classify what the user wants. Choose ONE of these categories:

1. **ANSWER** - The user is providing an answer to the current question
   Examples: "delhi", "v1.30", "calico", "4", "yes", "ubuntu", "general purpose"

2. **CHANGE** - The user wants to modify a previously selected parameter
   Examples: "I want to change the datacenter", "can we use a different zone", "go back to kubernetes version", "actually make it mumbai instead"

3. **OFF_TOPIC** - The user wants to do something completely different (not related to this cluster creation)
   Examples: "what clusters are in delhi", "show me load balancers", "list all VMs", "check firewall status"

4. **GREETING** - The user is greeting or asking about capabilities
   Examples: "hi", "hello", "what can you do", "help me", "who are you"

5. **CANCEL** - The user wants to stop/abort the cluster creation entirely
   Examples: "cancel", "stop this", "I don't want to create a cluster anymore", "abort"

**Important considerations:**
- If user says a city/location name like "delhi" or "mumbai" alone, it's likely an ANSWER (selecting datacenter)
- If user asks "what clusters are in delhi", it's OFF_TOPIC (querying existing clusters)
- If user mentions changing, modifying, or going back to something, it's CHANGE
- Brief acknowledgments or clarifications about the current question are ANSWER

Respond with a JSON object:
{"intent_type": "answer|change|off_topic|greeting|cancel", "details": "brief explanation", "change_target": "parameter name if CHANGE"}

2. Workflow Interruption Response Understanding
Location: _handle_workflow_interruption_response() method
Purpose: Understand user's choice when workflow is interrupted
The user was asked how to handle an interruption during cluster creation.
They were given these options:
- "abort" - Cancel cluster creation and handle their other request
- "continue" - Finish cluster creation first
- "save" - Save progress and handle other request, resume later

User's response: "{input_text}"

What did the user choose? Respond with ONLY one word: ABORT, CONTINUE, or SAVE
If unclear, respond with UNCLEAR.

3. Resume Intent Check
Location: _check_resume_intent() method
Purpose: Check if user wants to resume paused cluster creation
The user previously paused a cluster creation workflow.

User's message: "{input_text}"

Is the user trying to resume/continue the cluster creation they paused earlier?

Respond with ONLY: YES or NO

4. Criteria-Based Flavor Selection
Location: _select_flavor_by_criteria() method
Purpose: Select compute flavor based on user's criteria (e.g., "smallest", "at least 16GB")
You are helping select a compute flavor based on user criteria.

Available Flavors:
{flavors_str}

User's request: "{user_input}"

Analyze if the user is:
1. Using CRITERIA to select (e.g., "lowest", "smallest", "minimum", "cheapest", "least", "at least X", "around X")
2. Or just naming a specific option

If using CRITERIA:
- "lowest" / "smallest" / "minimum" / "cheapest" / "least" → Select the option with LOWEST resources (smallest vCPU, then smallest RAM)
- "highest" / "largest" / "maximum" / "most" → Select the option with HIGHEST resources
- "at least X vCPU" or "minimum X GB" → Select the SMALLEST option that meets the requirement
- "around X" / "approximately X" → Select the closest match

Respond with JSON:
- If criteria-based selection: {"criteria_based": true, "selected_index": <1-based index>, "reason": "brief explanation"}
- If NOT criteria-based (user naming specific option): {"criteria_based": false}