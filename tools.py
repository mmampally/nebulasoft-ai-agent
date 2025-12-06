import json
import os
from datetime import datetime
from typing import Type, ClassVar  # ← Add ClassVar here

from langchain.tools import BaseTool
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from pydantic import BaseModel, Field


# ============================================================
# Tool 1: DocSearchTool - Searches the Vector Database
# ============================================================

class DocSearchInput(BaseModel):
    """Input schema for DocSearchTool."""
    query: str = Field(description="The search query to find relevant documentation")


class DocSearchTool(BaseTool):
    name: str = "search_documentation"
    description: str = (
        "Searches the NebulaSoft technical documentation for information about "
        "error codes, setup instructions, features, and troubleshooting. "
        "Use this tool to answer technical support questions. "
        "Always cite the source document in your response."
    )
    args_schema: Type[BaseModel] = DocSearchInput
    
    def _run(self, query: str) -> str:
        """Execute the documentation search."""
        try:
            # Load embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name='sentence-transformers/all-MiniLM-L6-v2',
                model_kwargs={'device': 'cpu'}
            )
            
            # Load Chroma DB
            db = Chroma(
                persist_directory="./chroma_db",
                embedding_function=embeddings
            )
            
            # Search for relevant documents
            results = db.similarity_search_with_score(query, k=3)
            
            if not results:
                return "No relevant documentation was found for this query."
            
            # Format results with citations
            formatted_results = []
            for doc, score in results:
                source = doc.metadata.get('source', 'Unknown')
                chunk_id = doc.metadata.get('chunk_id', 'N/A')
                content = doc.page_content
                formatted_results.append(
                    f"[Source: {source}, Chunk: {chunk_id}, Relevance: {score:.2f}]\n{content}"
                )
            
            return "\n\n---\n\n".join(formatted_results)
            
        except Exception as e:
            return f"Error searching documentation: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        """Async version (not implemented for this project)."""
        raise NotImplementedError("Async not supported for this tool")


# ============================================================
# Tool 2: PricingCalculatorTool - Calculates Subscription Costs
# ============================================================

class PricingCalculatorInput(BaseModel):
    """Input schema for PricingCalculatorTool."""
    number_of_users: int = Field(description="Number of users for the subscription", gt=0)
    plan_type: str = Field(
        description="Type of plan: 'basic', 'pro', or 'enterprise'",
        pattern="^(basic|pro|enterprise)$"
    )


class PricingCalculatorTool(BaseTool):
    name: str = "calculate_pricing"
    description: str = (
        "Calculates the monthly subscription cost for NebulaSoft based on the number "
        "of users and plan type. Plan types are: 'basic' ($10/user), 'pro' ($20/user), "
        "or 'enterprise' ($40/user). Use this when customers ask about pricing or costs."
    )
    args_schema: Type[BaseModel] = PricingCalculatorInput
    
    # Pricing structure - mark as ClassVar
    PRICING: ClassVar[dict] = {  # ← Add ClassVar annotation
        "basic": 10,
        "pro": 20,
        "enterprise": 40
    }
    
    def _run(self, number_of_users: int, plan_type: str) -> str:
        """Calculate the subscription cost."""
        plan_type = plan_type.lower()
        
        if plan_type not in self.PRICING:
            return f"Error: Invalid plan type '{plan_type}'. Valid options are: basic, pro, enterprise"
        
        price_per_user = self.PRICING[plan_type]
        total_cost = number_of_users * price_per_user
        
        result = (
            f"Pricing Calculation:\n"
            f"Plan: {plan_type.capitalize()}\n"
            f"Number of Users: {number_of_users}\n"
            f"Price per User: ${price_per_user}/month\n"
            f"Total Monthly Cost: ${total_cost}/month"
        )
        
        return result
    
    async def _arun(self, number_of_users: int, plan_type: str) -> str:
        """Async version (not implemented for this project)."""
        raise NotImplementedError("Async not supported for this tool")


# ============================================================
# Tool 3: TicketEscalationTool - Files Support Tickets
# ============================================================

class TicketEscalationInput(BaseModel):
    """Input schema for TicketEscalationTool."""
    summary: str = Field(description="Brief summary of the issue")
    severity_level: str = Field(
        description="Severity level: 'low', 'medium', 'high'",
        pattern="^(low|medium|high)$"
    )


class TicketEscalationTool(BaseTool):
    name: str = "escalate_ticket"
    description: str = (
        "Files a support ticket for escalation to Tier-2 support. "
        "Use this tool ONLY when the documentation search cannot answer the user's question "
        "or when the issue requires human intervention. "
        "Severity levels: 'low', 'medium', 'high'."
    )
    args_schema: Type[BaseModel] = TicketEscalationInput
    
    def _run(self, summary: str, severity_level: str) -> str:
        """File a support ticket."""
        severity_level = severity_level.lower()
        
        if severity_level not in ["low", "medium", "high"]:
            return f"Error: Invalid severity level '{severity_level}'"
        
        # Create ticket data
        ticket = {
            "ticket_id": f"TKT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "severity": severity_level,
            "status": "open"
        }
        
        # Append to tickets.log
        try:
            with open("tickets.log", "a", encoding="utf-8") as f:
                f.write(json.dumps(ticket) + "\n")
            
            result = (
                f"🎫 Ticket Escalated Successfully!\n"
                f"Ticket ID: {ticket['ticket_id']}\n"
                f"Severity: {severity_level.upper()}\n"
                f"Summary: {summary}\n"
                f"Status: Open\n"
                f"A Tier-2 support representative will contact you shortly."
            )
            return result
            
        except Exception as e:
            return f"Error filing ticket: {str(e)}"
    
    async def _arun(self, summary: str, severity_level: str) -> str:
        """Async version (not implemented for this project)."""
        raise NotImplementedError("Async not supported for this tool")


# ============================================================
# Convenience function to get all tools
# ============================================================

def get_all_tools():
    """Returns a list of all available tools."""
    return [
        DocSearchTool(),
        PricingCalculatorTool(),
        TicketEscalationTool()
    ]