
import google.generativeai as genai
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import logging
from typing import Optional


class PolicyRAGSystem:
    def __init__(self, retriever, config, data_loader=None):
        self.retriever = retriever
        self.data_loader = data_loader
        self.llm = None
        self.chain = None
        self.config = config
        self.llm_ready = False

        logging.basicConfig(
            filename='rag_logs.txt',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        genai.configure(api_key=self.config.GEMINI_API_KEY)

    def initialize_llm(self) -> None:
        try:
            self.llm = genai.GenerativeModel(
                model_name=self.config.GEMINI_MODEL,
                generation_config={
                    "temperature": self.config.GEMINI_TEMPERATURE,
                    "max_output_tokens": 2000
                }
            )
            self.llm_ready = True
            logging.info("Gemini LLM initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize Gemini: {str(e)}")
            raise Exception(f"Failed to initialize Gemini: {str(e)}")

    def custom_retriever(self, query):
        if self.data_loader:
            reranked_chunks = self.data_loader.get_reranked_chunks(query, top_k=5)
            context = "\n\n".join([chunk.page_content for chunk in reranked_chunks])
            logging.info(f"Custom reranking applied for query: {query}")
            logging.info(f"Retrieved {len(reranked_chunks)} reranked chunks")
            return context
        else:
            docs = self.retriever.invoke(query)
            return "\n\n".join([doc.page_content for doc in docs])

    def setup_qa_chain(self) -> None:
        if not self.llm_ready or self.retriever is None:
            raise Exception("LLM or retriever not initialized")

        template = """You are a highly efficient and concise Bank Policy Assistant for [Bank Name].
Your primary role is to answer employee questions about internal bank policies,
strictly based on the official Bank Policy Manual.

Context: {context}

Question: {question}

## 🔒 STRICT INSTRUCTIONS FOR RESPONDING:

1. EXTREME CONCISENESS REQUIRED:
   - Limit your answer to 2–4 plain sentences maximum.
   - Do not use bullets, numbering, or markdown formatting.
   - Provide only the direct answer — avoid pleasantries, summaries, or restatements of the question.

2. STRICTLY FROM CONTEXT ONLY:
   - Your response must be based only on the provided policy context.
   - Do not infer or assume anything not explicitly present in the context.

3. SYNONYM AND RELATED TERM IDENTIFICATION:
   If the exact term from the question is not in the context, search for related terms and synonyms. For example:
   - "fuel", "petrol", "transport", "commute" → "Travel Allowance"
   - "loan", "advance", "finance" → "Loan Policy / Advance Salary"
   - "medical", "insurance", "health" → "Medical Benefits / OPD Policy"
   - "vacation", "PTO", "leave", "holiday" → "Leave Policy"
   - "bonus", "commission", "incentive" → "Performance Incentives"
   - "termination", "resignation", "exit", "job end" → "Exit Policy"
   - "bond", "contract", "non-compete" → "Employment Bond / Non-Competing Clause"

4. TRIM DOWN EXCESSIVE DETAIL:
   If the context contains long or multi-part explanations, extract and summarize only the parts directly answering the question. Skip unrelated content.

5. NO INFERRED OR EXTERNAL INFORMATION:
   Never guess, infer, or fabricate. Stick strictly to what’s explicitly stated in the context or via synonym mapping.

6. NO IRRELEVANT DETAILS:
   Avoid any content that does not directly answer the question. Your job is to filter out noise.

7. FALLBACK RESPONSES (Use ONLY if needed):
   - If no relevant policy is found: "According to the current bank policy, this benefit/policy is not available."
   - If the question concerns personal records: "This requires review of your personal employment record. Please contact HR or your manager."
   - If the situation involves exceptions or management discretion: "This situation may require management approval. Please check with your department head."
   - If it's completely out of scope: "I don't have this information in the available bank policies. Please contact HR for assistance."

8. HANDLING RUDE OR ABUSIVE LANGUAGE:
   - If the user's question contains offensive, aggressive, or abusive language (e.g., insults, profanity), do not answer the question.
   - Instead, respond with: "😕 Let's keep it respectful. I'm here to help you. Please relax and rephrase your question."

9. HANDLING FRIENDLY OR SMALL-TALK MESSAGES:
   - If the user asks how you are, compliments you, or says things like “love you”, “you're great”, “thank you”, etc., respond briefly and kindly.
   - Example: “Thanks for the kind words! I'm here to help with policy questions — feel free to ask.”
   - Always follow up with a gentle nudge to ask a policy-related question.

10. HANDLING IRRELEVANT OR RANDOM QUERIES:
   - If the user's query seems unrelated to bank policies (e.g., random facts, jokes, news, non-work topics), respond with:
     "I'm here to help with bank policy-related questions. Please ask something related to internal policies or employment."

Your answer must be accurate, minimal, and based only on the provided context and user grade.
"""


        prompt = ChatPromptTemplate.from_template(template)

        self.chain = (
            {
                "context": lambda x: self.custom_retriever(x["question"]),
                "question": lambda x: x["question"]
                
            }
            | prompt
            | self._invoke_gemini
        )

    def _invoke_gemini(self, prompt) -> str:
        try:
            prompt_str = prompt.to_string()
            response = self.llm.generate_content(
                prompt_str,
                generation_config={
                    "temperature": self.config.GEMINI_TEMPERATURE,
                    "max_output_tokens": 800
                }
            )
            result = response.text.strip()

            if result.count(". ") > 4 or len(result) > 600:
                logging.info("Summarizing Gemini response for conciseness...")
                summarization_prompt = (
                    "Summarize the following text into 2–4 concise natural sentences, "
                    "preserving all key HR policy details, without using bullets or headings:\n\n" + result
                )
                summary_response = self.llm.generate_content(
                    summarization_prompt,
                    generation_config={
                        "temperature": 0.3,
                        "max_output_tokens": 400
                    }
                )
                summarized = summary_response.text.strip()
                logging.info(f"Summarized response: {summarized}")
                return summarized
            else:
                return result

        except Exception as e:
            logging.error(f"Gemini API error: {str(e)}")
            return f"Error processing your query: {str(e)}"

    def query_policy(self, question: str, user_grade: Optional[str] = None) -> str:
        if not self.chain:
            return "System initializing, please wait..."

        try:
            logging.info(f"Processing query with custom reranking: {question}")
            response = self.chain.invoke({
                "question": question
                
            })
            logging.info(f"Response: {response}")
            return response
        except Exception as e:
            logging.error(f"Query processing error: {str(e)}")
            return f"Error processing your query: {str(e)}"
