prompt = "Please select an engagement to work with:\n\n"
        for opt in options:
            prompt += f"**{opt['index']}. {opt['name']}** (ID: {opt['id']})\n"
        prompt += "\nYou can say the number, name, or ID. You can also change this later by saying 'switch engagement'."
        
        return {
            "engagements": engagements,
            "options": options,
            "prompt": prompt,
            "needs_selection": True
        }