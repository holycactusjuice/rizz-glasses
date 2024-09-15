from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from templates import SUGGESTION_TEMPLATE_DATE, SCORING_TEMPLATE_DATE, SUMMARIZE_TEMPLATE_DATE, SUGGESTION_TEMPLATE_BUSINESS, SCORING_TEMPLATE_BUSINESS, SUMMARIZE_TEMPLATE_BUSINESS

class GroqConversationAnalyzer:
    def __init__(self, api_key, user):

        # Initialize LangChain components
        self.llm = ChatGroq(api_key=api_key)
        self.user = user
        self.memory = ConversationBufferMemory(input_key="human_input", memory_key="chat_history")
        self.template_date_suggestion = PromptTemplate(
            input_variables=["chat_history, human_input"],
            template=SUGGESTION_TEMPLATE_DATE
        )
        self.template_date_scoring = PromptTemplate(
        input_variables=["input", "chat_history"],
        template=SCORING_TEMPLATE_DATE
        )
        self.template_date_summarize = PromptTemplate(
        input_variables=["human_input","chat_history"],
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
        input_variables=["human_input","chat_history"],
        template=SUMMARIZE_TEMPLATE_BUSINESS
        )

        
        self.chain_suggestion_date = LLMChain(llm=self.llm, prompt=self.template_date_suggestion, memory=self.memory)
        self.chain_score_date = LLMChain(llm=self.llm, prompt=self.template_date_scoring, memory=self.memory)
        self.chain_summarize_date = LLMChain(llm=self.llm, prompt=self.template_date_summarize, memory=self.memory)

        self.chain_suggestion_business = LLMChain(llm=self.llm, prompt=self.template_business_suggestion, memory=self.memory)
        self.chain_score_business = LLMChain(llm=self.llm, prompt=self.template_business_scoring, memory=self.memory)
        self.chain_summarize_business = LLMChain(llm=self.llm, prompt=self.template_business_summarize, memory=self.memory)

    
    def return_suggestion(self, type="date"):
        if type == "date":
            chat_history = self.memory.load_memory_variables({})["chat_history"]
            response = self.chain_suggestion_date.invoke({"human_input": "", "chat_history": chat_history})
            suggestion = response['text'].strip()
            return suggestion
        elif type == "business":
            chat_history = self.memory.load_memory_variables({})["chat_history"]
            response = self.chain_suggestion_business.invoke({"human_input": "", "chat_history": chat_history})
            suggestion = response['text'].strip()
            return suggestion

    def score_sentence(self, sentence, type="date"):
        if type == "date":
            chat_history = self.memory.load_memory_variables({})["chat_history"]
            response = self.chain_score_date.invoke({"human_input": sentence, "chat_history": chat_history})
        elif type == "business":
            chat_history = self.memory.load_memory_variables({})["chat_history"]
            response = self.chain_score_business.invoke({"human_input": sentence, "chat_history": chat_history})
        # Parse the response to extract score and explanation
        lines = response['text'].strip().split('\n')
        score = int(lines[0].split(':')[1].strip())
        explanation = '\n'.join(lines[1:])
        
        return score, explanation

    def process_input(self, input_text):
        sentences = input_text.split('.')  # Simple sentence splitting
        results = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:  # Skip empty sentences
                score, explanation = self.score_sentence(sentence)
                results.append({
                    'sentence': sentence,
                    'score': score,
                    'explanation': explanation
                })
                self.memory.save_context({"human_input": input_text}, {"output": ""})
        
        return results
    
    def summarize_conversation(self, type="date"):
        if type == "date":
            chat_history = self.memory.load_memory_variables({})["chat_history"]
            response = self.chain_summarize_date.invoke({"human_input": "", "chat_history": chat_history})
            return response['text']
        elif type == "business":
            chat_history = self.memory.load_memory_variables({})["chat_history"]
            response = self.chain_summarize_business.invoke({"human_input": "", "chat_history": chat_history})
            return response['text']

    def run_conversation(self, input_text, terminal=True):
        if terminal:
            while True:
                input_text = input("\nYou: ")
                if input_text.lower() == 'end':
                    print(self.summarize_conversation())
                    break
                elif input_text.lower() == 'return_suggestion':
                    suggestion = self.return_suggestion()
                    print(f"{suggestion}")
                    continue
                
                results = self.process_input(input_text)
                for result in results:
                    print(f"\nSentence: {result['sentence']}")
                    print(f"Score: {result['score']}")
                    print(f"{result['explanation']}")


if __name__ == "__main__":
    api_key = "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"
    analyzer = GroqConversationAnalyzer(api_key=api_key, user=True)
    analyzer.run_conversation("hello", terminal=True)