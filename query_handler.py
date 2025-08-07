from typing import Dict, Optional
from datetime import datetime
import streamlit as st

class QueryHandler:
    def __init__(self, authenticated_user: Dict, rag_system, authenticator):
        self.user = authenticated_user
        self.rag_system = rag_system
        self.authenticator = authenticator
        self.hr_contact = "hr@bankname.com"

    def _refresh_user_data(self):
        """Refresh user data from Google Sheets and update session state"""
        try:
            updated_user = self.authenticator.get_authenticated_user()
            if updated_user:
                self.user = updated_user
                st.session_state.user = updated_user
                return True
        except Exception as e:
            st.warning(f"Could not refresh user data: {str(e)}")
            return False

    def handle_query(self, query: str) -> str:
        """Process user queries with semantic understanding"""
        query_lower = query.lower().strip()

        # Handle leave application explicitly
        if query_lower.startswith("apply for leave"):
            return self._handle_leave_application(query_lower)

        # Handle leave balance queries
        if any(term in query_lower for term in ["leave balance", "remaining leaves", "how many leaves", "my leaves"]):
            self._refresh_user_data()
            return f"💼 Your current leave balance: **{self.user.get('remaining_leaves', 'N/A')} days**"

        # Handle greetings and casual queries
        if any(term in query_lower for term in ["hello", "hi", "hey", "good morning", "good afternoon"]):
            return f"Hello {self.user['username']}! 👋 I'm here to help you with bank policy questions and leave applications. What can I assist you with today?"

        # Handle help queries
        if any(term in query_lower for term in ["help", "what can you do", "how to use", "commands"]):
            return self._get_help_response()

        # Handle thank you
        if any(term in query_lower for term in ["thank you", "thanks", "appreciate"]):
            return "You're welcome! 😊 Feel free to ask me anything else about bank policies or leave applications."

        # Otherwise: treat it as a semantic HR policy query
        try:
            response = self.rag_system.query_policy(query, self.user.get('grade'))

            # Retry once if empty/null
            if not response or not response.strip():
                response = self.rag_system.query_policy(query, self.user.get('grade'))

            # If still nothing, return fallback message
            if not response or not response.strip():
                return (
                    "🤔 I couldn't find a clear answer to your question in the bank policies. "
                    f"Please contact HR at **{self.hr_contact}** for assistance, or try rephrasing your question."
                )

            return self._refine_policy_response(response)

        except Exception as e:
            return f"⚠️ Sorry, there was an error processing your question: {str(e)}"

    def _handle_leave_application(self, query: str) -> str:
        """Process leave application"""
        try:
            # Parse leave days from query
            parts = query.split()
            days = None
            
            # Try to find the number in the query
            for part in parts:
                try:
                    days = float(part)
                    break
                except ValueError:
                    continue

            if not days:
                return "❌ Please specify leave days like: 'apply for leave 2.5' or 'apply for leave 1'"

            # Apply for leave
            success = self.authenticator.apply_for_leave(self.user['username'], days)
            if success:
                self._refresh_user_data()
                return (
                    f"✅ **Leave Application Submitted Successfully!**\n\n"
                    f"📊 **Details:**\n"
                    f"• Applied for: **{days} days**\n"
                    f"• Remaining leaves: **{self.user.get('remaining_leaves', 'N/A')} days**\n"
                    f"• Status: **Pending Approval** ⏳\n\n"
                    f"You will receive confirmation once your application is reviewed."
                )
            else:
                return "❌ Leave application failed. Please try again or contact HR."
                
        except ValueError as e:
            return f"❌ **Application Error:** {str(e)}"
        except Exception as e:
            return f"❌ **Failed to apply for leave:** {str(e)}"

    def _refine_policy_response(self, response: Optional[str]) -> str:
        """Refine policy responses with user-specific info"""
        if not response or not response.strip():
            return (
                "🤔 I couldn't find a clear answer to your question in the bank policies. "
                f"Please contact HR at **{self.hr_contact}** for assistance."
            )

        # Add user-specific leave balance if the response mentions leave
        if "leave" in response.lower() and "remaining_leaves" in self.user:
            response += f"\n\n💼 **Your current leave balance:** {self.user['remaining_leaves']} days"

        return response.strip()

    def _get_help_response(self) -> str:
        """Provide help information to the user"""
        return (
            "🤖 **I'm your Bank Policy Assistant!** Here's what I can help you with:\n\n"
            "**📋 Policy Questions:**\n"
            "• Medical/Health policies\n"
            "• Travel and transport allowances\n"
            "• Loan and advance policies\n"
            "• Performance bonuses\n"
            "• Exit procedures\n\n"
            "**🌿 Leave Management:**\n"
            "• Check your leave balance\n"
            "• Apply for leave (e.g., 'apply for leave 2')\n"
            "• Leave policy information\n\n"
            "**💡 Examples:**\n"
            "• 'What is the medical policy?'\n"
            "• 'How do I get travel allowance?'\n"
            "• 'Apply for leave 1.5'\n"
            "• 'What is my leave balance?'\n\n"
            "Just ask me anything in natural language! 😊"
        )

    def _get_default_response(self) -> str:
        """Default response for unrecognized queries"""
        return (
            "🤔 I'm not sure I understand your question. I can help with:\n\n"
            "• **Bank policy questions** (medical, travel, loans, etc.)\n"
            "• **Leave applications and balances**\n"
            "• **HR procedures and guidelines**\n\n"
            "Try rephrasing your question or type 'help' for more examples!"
        )