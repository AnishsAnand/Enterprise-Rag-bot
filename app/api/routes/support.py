from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import json

router = APIRouter()

class SupportTicket(BaseModel):
    name: str
    email: EmailStr
    type: str
    priority: str
    subject: str
    description: str
    includeSystemInfo: bool = True
    systemInfo: Optional[Dict[str, Any]] = None

class ChatMessage(BaseModel):
    message: str

class SupportArticle(BaseModel):
    id: str
    title: str
    content: str
    category: str
    tags: List[str]
    helpful_count: int

SUPPORT_ARTICLES = [
    {
        "id": "1",
        "title": "Getting Started with Enterprise RAG Bot",
        "content": """
        <p>Welcome to Enterprise RAG Bot! Here's how to get started:</p>
        <ol>
            <li><strong>Set up your API keys:</strong> Configure OpenRouter.ai, Voyage, and Ollama credentials</li>
            <li><strong>Start scraping:</strong> Use the Web Scraper to collect data from websites</li>
            <li><strong>Query your data:</strong> Use the RAG System to ask questions about scraped content</li>
            <li><strong>Monitor system:</strong> Check the Admin panel for system health and metrics</li>
        </ol>
        <p>For detailed setup instructions, check our documentation or contact support.</p>
        """,
        "category": "Getting Started",
        "tags": ["setup", "quickstart", "tutorial"],
        "helpful_count": 45
    },
    {
        "id": "2",
        "title": "Web Scraping Best Practices",
        "content": """
        <p>To get the best results from web scraping:</p>
        <ul>
            <li><strong>Respect robots.txt:</strong> Always check the website's robots.txt file</li>
            <li><strong>Use delays:</strong> Add appropriate delays between requests</li>
            <li><strong>Handle errors:</strong> Implement proper error handling and retries</li>
            <li><strong>Monitor performance:</strong> Keep track of success rates and response times</li>
        </ul>
        <p>Our system includes built-in anti-blocking mechanisms to help you scrape responsibly.</p>
        """,
        "category": "Web Scraping",
        "tags": ["scraping", "best-practices", "ethics"],
        "helpful_count": 32
    },
    {
        "id": "3",
        "title": "Understanding RAG System Results",
        "content": """
        <p>The RAG (Retrieval-Augmented Generation) system works by:</p>
        <ol>
            <li><strong>Document Retrieval:</strong> Finding relevant documents based on your query</li>
            <li><strong>Context Assembly:</strong> Combining relevant information from multiple sources</li>
            <li><strong>AI Generation:</strong> Using AI to generate accurate, contextual responses</li>
        </ol>
        <p>Results include source citations and relevance scores to help you verify information.</p>
        """,
        "category": "RAG System",
        "tags": ["rag", "ai", "retrieval", "generation"],
        "helpful_count": 28
    },
    {
        "id": "4",
        "title": "API Rate Limits and Usage",
        "content": """
        <p>Understanding API limits:</p>
        <ul>
            <li><strong>OpenRouter.ai:</strong> Varies by model and subscription</li>
            <li><strong>Voyage AI:</strong> Check your dashboard for current limits</li>
            <li><strong>Local Ollama:</strong> Limited by your hardware resources</li>
        </ul>
        <p>The system automatically handles fallbacks when limits are reached.</p>
        """,
        "category": "API Usage",
        "tags": ["api", "limits", "usage", "billing"],
        "helpful_count": 19
    },
    {
        "id": "5",
        "title": "Troubleshooting Common Issues",
        "content": """
        <p>Common issues and solutions:</p>
        <ul>
            <li><strong>Scraping fails:</strong> Check target website accessibility and anti-bot measures</li>
            <li><strong>AI responses are poor:</strong> Verify API keys and check service status</li>
            <li><strong>ChromaDB errors:</strong> Ensure proper file permissions and disk space</li>
            <li><strong>Performance issues:</strong> Monitor system resources and adjust concurrency</li>
        </ul>
        <p>Check the Admin panel for detailed system logs and metrics.</p>
        """,
        "category": "Troubleshooting",
        "tags": ["troubleshooting", "errors", "debugging"],
        "helpful_count": 41
    }
]

CHAT_HISTORY = []

@router.post("/ticket")
async def submit_support_ticket(ticket: SupportTicket):
    """Submit a support ticket"""
    try:
        ticket_id = str(uuid.uuid4())[:8].upper()
        
        
        ticket_data = {
            "ticket_id": ticket_id,
            "timestamp": datetime.now().isoformat(),
            "status": "open",
            **ticket.dict()
        }
        
        print(f"Support ticket created: {ticket_id}")
        print(f"Ticket data: {json.dumps(ticket_data, indent=2)}")
        
        return {
            "status": "success",
            "ticket_id": ticket_id,
            "message": "Support ticket submitted successfully. You will receive a confirmation email shortly.",
            "estimated_response_time": "24 hours"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/articles")
async def get_support_articles():
    """Get all support articles"""
    try:
        return {
            "status": "success",
            "articles": SUPPORT_ARTICLES,
            "total_count": len(SUPPORT_ARTICLES)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/articles/search")
async def search_support_articles(query: str):
    """Search support articles"""
    try:
        if not query.strip():
            return {
                "status": "success",
                "articles": SUPPORT_ARTICLES,
                "query": query,
                "total_count": len(SUPPORT_ARTICLES)
            }
        
        query_lower = query.lower()
        filtered_articles = []
        
        for article in SUPPORT_ARTICLES:
            if (query_lower in article["title"].lower() or
                query_lower in article["content"].lower() or
                query_lower in article["category"].lower() or
                any(query_lower in tag.lower() for tag in article["tags"])):
                filtered_articles.append(article)
        
        return {
            "status": "success",
            "articles": filtered_articles,
            "query": query,
            "total_count": len(filtered_articles)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/message")
async def send_chat_message(message: ChatMessage):
    """Send a chat message and get AI response"""
    try:
        user_message = message.message.strip()
        user_msg = {
            "id": str(uuid.uuid4()),
            "message": user_message,
            "sender": "user",
            "timestamp": datetime.now().isoformat(),
            "type": "text"
        }
        CHAT_HISTORY.append(user_msg)
        ai_response = generate_support_response(user_message)
        ai_msg = {
            "id": str(uuid.uuid4()),
            "message": ai_response,
            "sender": "agent",
            "timestamp": datetime.now().isoformat(),
            "type": "text"
        }
        CHAT_HISTORY.append(ai_msg)
        
        return {
            "status": "success",
            "message": ai_response,
            "sender": "agent",
            "timestamp": ai_msg["timestamp"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chat/history")
async def get_chat_history():
    """Get chat message history"""
    try:
        return {
            "status": "success",
            "messages": CHAT_HISTORY[-20:],  
            "total_count": len(CHAT_HISTORY)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def generate_support_response(message: str) -> str:
    """Generate contextual support responses"""
    message_lower = message.lower()

    if any(word in message_lower for word in ['start', 'begin', 'setup', 'install', 'configure']):
        return """Hi! I'd be happy to help you get started with Enterprise RAG Bot. Here are the key steps:

1. **Set up API keys**: Configure your OpenRouter.ai, Voyage, and Ollama credentials in the .env file
2. **Start the services**: Run both the FastAPI backend and Angular frontend
3. **Test scraping**: Try scraping a simple website to verify everything works
4. **Explore RAG**: Upload some documents and test the query functionality

Would you like detailed instructions for any of these steps?"""

    elif any(word in message_lower for word in ['error', 'problem', 'issue', 'bug', 'broken', 'not working']):
        return """I'm sorry to hear you're experiencing issues. To help you better, could you please provide:

1. **What were you trying to do?** (scraping, querying, etc.)
2. **What error message did you see?** (if any)
3. **When did this start happening?**
4. **Have you made any recent changes?**

In the meantime, you can check:
- System status in the Admin panel
- Browser console for any JavaScript errors
- Backend logs for detailed error information

I'm here to help resolve this quickly!"""

    elif any(word in message_lower for word in ['scrap', 'crawl', 'extract', 'website', 'data']):
        return """Great question about web scraping! Our system offers several powerful features:

**Single URL Scraping:**
- Extract text, links, images, and tables
- Handle dynamic content with Selenium
- Multiple output formats (JSON, CSV, text, PDF)

**Bulk Scraping:**
- Automatic URL discovery from base domains
- Configurable crawl depth and limits
- Built-in anti-blocking mechanisms

**Best Practices:**
- Respect robots.txt and rate limits
- Use appropriate delays between requests
- Monitor success rates in the Admin panel

What specific scraping task are you working on? I can provide more targeted guidance!"""

    elif any(word in message_lower for word in ['rag', 'query', 'search', 'ai', 'question', 'answer']):
        return """The RAG (Retrieval-Augmented Generation) system is one of our most powerful features! Here's how it works:

**Document Storage:**
- Automatically stores scraped content in ChromaDB
- Supports manual document upload
- Handles multiple file formats

**Intelligent Retrieval:**
- Uses vector similarity search to find relevant content
- Combines information from multiple sources
- Provides relevance scores and source citations

**AI-Powered Responses:**
- Multiple AI providers (OpenRouter, Voyage, Ollama)
- Automatic fallback if one service is unavailable
- Context-aware responses based on your data

Try asking a question about your scraped content - the system will find relevant information and generate a comprehensive answer!"""

    elif any(word in message_lower for word in ['api', 'key', 'billing', 'cost', 'limit', 'quota']):
        return """Here's what you need to know about API usage and costs:

**Required API Keys:**
- **OpenRouter.ai**: Primary AI service (pay-per-use)
- **Voyage AI**: Embedding generation (subscription-based)
- **Ollama**: Local AI (free, but requires local setup)

**Cost Management:**
- The system automatically uses the most cost-effective option
- Ollama provides free local processing as backup
- Monitor usage in each provider's dashboard

**Rate Limits:**
- Automatic handling of rate limits
- Intelligent fallback between providers
- Configurable request delays

Need help setting up any specific API keys or have questions about pricing?"""

    elif any(word in message_lower for word in ['help', 'support', 'how', 'what', 'guide']):
        return """I'm here to help! Here are some ways I can assist you:

**Quick Help:**
- Getting started guide and setup
- Troubleshooting technical issues
- Best practices for web scraping
- Understanding RAG system results
- API configuration and usage

**Resources:**
- Check our Help Articles tab for detailed guides
- Visit the Admin panel for system status
- Contact form for complex issues

**Popular Topics:**
- Setting up API keys
- Web scraping best practices
- RAG system optimization
- Troubleshooting common errors

What specific area would you like help with? Just let me know and I'll provide detailed guidance!"""

   
    else:
        return """Thank you for reaching out! I want to make sure I give you the most helpful response possible.

Could you please provide a bit more detail about:
- What you're trying to accomplish
- Any specific issues you're encountering
- Which part of the system you're working with (scraping, RAG, admin, etc.)

I'm here to help with:
1)Setup and configuration
2)Web scraping guidance  
3) RAG system optimization
4)Troubleshooting issues
5) Best practices and tips

Feel free to ask me anything about Enterprise RAG Bot!"""
