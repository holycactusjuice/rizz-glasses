from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from templates import SUGGESTION_TEMPLATE_DATE, SCORING_TEMPLATE_DATE, SUMMARIZE_TEMPLATE_DATE, SUGGESTION_TEMPLATE_BUSINESS, SCORING_TEMPLATE_BUSINESS, SUMMARIZE_TEMPLATE_BUSINESS
from dotenv import load_dotenv
import os


class GroqConversationAnalyzer:
    def __init__(self, api_key):

        # Initialize LangChain components
        self.llm = ChatGroq(api_key=api_key)
        self.memory = ConversationBufferMemory(
            input_key="human_input", memory_key="chat_history")
        self.template_date_suggestion = PromptTemplate(
            input_variables=["chat_history, human_input"],
            template=SUGGESTION_TEMPLATE_DATE
        )
        self.template_date_scoring = PromptTemplate(
            input_variables=["input", "chat_history"],
            template=SCORING_TEMPLATE_DATE
        )
        self.template_date_summarize = PromptTemplate(
            input_variables=["human_input", "chat_history"],
            template=SUMMARIZE_TEMPLATE_DATE
        )

        self.template_business_suggestion = PromptTemplate(
            input_variables=["chat_history, human_input"],
            template=SUGGESTION_TEMPLATE_BUSINESS
        )
        self.template_business_scoring = PromptTemplate(
            input_variables=["input", "chat_history"],
            template=SCORING_TEMPLATE_BUSINESS
        )
        self.template_business_summarize = PromptTemplate(
            input_variables=["human_input", "chat_history"],
            template=SUMMARIZE_TEMPLATE_BUSINESS
        )

        self.chain_suggestion_date = LLMChain(
            llm=self.llm, prompt=self.template_date_suggestion, memory=self.memory)
        self.chain_score_date = LLMChain(
            llm=self.llm, prompt=self.template_date_scoring, memory=self.memory)
        self.chain_summarize_date = LLMChain(
            llm=self.llm, prompt=self.template_date_summarize, memory=self.memory)

        self.chain_suggestion_business = LLMChain(
            llm=self.llm, prompt=self.template_business_suggestion, memory=self.memory)
        self.chain_score_business = LLMChain(
            llm=self.llm, prompt=self.template_business_scoring, memory=self.memory)
        self.chain_summarize_business = LLMChain(
            llm=self.llm, prompt=self.template_business_summarize, memory=self.memory)

    def return_suggestion(self, type="date"):
        if type == "date":
            chat_history = self.memory.load_memory_variables({})[
                "chat_history"]
            response = self.chain_suggestion_date.invoke(
                {"human_input": "", "chat_history": chat_history})
            suggestion = response['text'].strip()
            return suggestion
        elif type == "business":
            chat_history = self.memory.load_memory_variables({})[
                "chat_history"]
            response = self.chain_suggestion_business.invoke(
                {"human_input": "", "chat_history": chat_history})
            suggestion = response['text'].strip()
            return suggestion

    def score_sentence(self, sentence, type="date"):
        if type == "date":
            chat_history = self.memory.load_memory_variables({})[
                "chat_history"]
            response = self.chain_score_date.invoke(
                {"human_input": sentence, "chat_history": chat_history})
        elif type == "business":
            chat_history = self.memory.load_memory_variables({})[
                "chat_history"]
            response = self.chain_score_business.invoke(
                {"human_input": sentence, "chat_history": chat_history})
        # Parse the response to extract score and explanation
        lines = response['text'].strip().split('\n')
        score = int(lines[0].split(':')[1].strip())
        explanation = '\n'.join(lines[1:])

        return score, explanation

    def process_input(self, input_text, type="date"):
        sentences = input_text.split('.')
        results = []

        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                score, explanation = self.score_sentence(sentence, type)
                results.append({
                    'sentence': sentence,
                    'score': score,
                    'explanation': explanation
                })
                self.memory.save_context(
                    {"human_input": input_text}, {"output": ""})

        return results

    def summarize_conversation(self, type="date"):
        if type == "date":
            chat_history = self.memory.load_memory_variables({})[
                "chat_history"]
            response = self.chain_summarize_date.invoke(
                {"human_input": "", "chat_history": chat_history})
            return response['text']
        elif type == "business":
            chat_history = self.memory.load_memory_variables({})[
                "chat_history"]
            response = self.chain_summarize_business.invoke(
                {"human_input": "", "chat_history": chat_history})
            return response['text']

        return results

    def summarize_conversation(self, type="date"):
        template = "Summarize the following conversation in the context of a {}: {}".format(
            "date" if type == "date" else "business meeting",
            self.conversation_history
        )
        messages = [{"role": "system", "content": template}]
        return self._execute_with_retry(messages)

    def run_conversation(self, input_text, terminal=True, type="date"):
        if terminal:
            while True:
                input_text = input("\nYou: ")
                if input_text.lower() == 'end':
                    print(self.summarize_conversation(type))
                    break
                elif input_text.lower() == 'return_suggestion':
                    suggestion = self.return_suggestion(type)
                    print(f"Suggestion: {suggestion}")
                    continue

                results = self.process_input(input_text, type)
                for result in results:
                    print(f"\nSentence: {result['sentence']}")
                    print(f"Score: {result['score']}")
                    print(f"Explanation: {result['explanation']}")


def main():
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not found in environment variables.")
        return

    try:
        analyzer = GroqConversationAnalyzer(api_key=api_key)
        analyzer.run_conversation("hello", terminal=True, type="date")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    load_dotenv()
    api_key = "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"
    analyzer = GroqConversationAnalyzer(api_key=api_key)
    analyzer.run_conversation("hello", terminal=True)
