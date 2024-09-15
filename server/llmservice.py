from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory

class GroqConversationAnalyzer:
    def __init__(self, api_key, user):

        # Initialize LangChain components
        self.llm = ChatGroq(api_key=api_key)
        self.user = user
        self.memory = ConversationBufferMemory(input_key="human_input", memory_key="chat_history")
        self.prompt_template_suggestion = PromptTemplate(
            input_variables=["chat_history, human_input"],
            template=
        """
        {human_input}
        You are a first date, dating conversation analyst chatbot, and you are tasked with providing a suggestion for the next sentence in a first date between two individuals.
        Given the following conversation history, suggest the next sentence that would be appropriate, engaging and expressive of the person in the conversation.
        Use past context to suggest the timing of the sentence. Some sentences become okay to say later on in the conversation.

        Chat History:
        {chat_history}


        Suggestion: [Insert your suggested sentence here]
        Explanation: [Provide a brief explanation of why you decided to choose this sentence]
        """
        )
 
        self.prompt_template_scoring = PromptTemplate(
        input_variables=["input", "chat_history"],
        template=
        """
        You are an expert in social dynamics and dating conversations. Your task is to analyze and rate sentences within the context of a first date conversation, considering the progression and build-up of the interaction.

        Given the conversation history, the new input sentence, and the current turn number in the conversation, give the sentence a score between 1-10 based on the following criteria:

        1. Appropriateness: How suitable is this sentence for the current stage of the conversation?
        2. Engagement: How well does this sentence contribute to an interesting and flowing conversation?
        3. Expressiveness: How effectively does this sentence convey the speaker's personality or interests?
        4. Timing: Considering the conversation's progression, how well-timed is this sentence?
        5. Context: How well does this sentence fit with the established rapport and emotional connection?

        Remember:
        - The appropriateness of certain statements, including expressions of strong feelings, may increase as the conversation progresses and rapport builds.
        - Consider the balance between openness and maintaining appropriate boundaries, which may shift as the conversation develops.
        - Evaluate how well the sentence builds upon or relates to previous parts of the conversation.
        - Think about how the sentence might impact the comfort level and rapport between the individuals at this stage of the interaction.

        Conversation History:
        {chat_history}

        New Input Sentence: {human_input}

        Score: [Insert a single integer between 1 and 10]
        Brief Explanation: [Offer a concise analysis of the sentence's effectiveness, appropriateness, and impact on the conversation, considering the current stage of the interaction]
        """
        )

        self.prompt_template_summarize = PromptTemplate(
        input_variables=["human_input","chat_history"],
        template=
    """
    {human_input}
    You are an expert in analyzing dating conversations, particularly first dates. Your task is to provide a concise summary of the following conversation between a user and their date, focusing on the content, what went well, and what did not go well.

    Chat History:
    {chat_history}

    Please provide a complete summary overview connecting the following sentence into a coherent paragraph:

    1. Content Overview:
    [Provide a brief overview of the main topics discussed and the general flow of the conversation]

    2. Positive Aspects:
    [List 2-3 things that went well in the conversation, such as good rapport building, shared interests, or effective communication]

    3. Areas for Improvement:
    [Identify 2-3 aspects of the conversation that could have been better, such as missed opportunities, awkward moments, or potential misunderstandings]

    4. Key Insights:
    [Provide 1-2 key insights about the overall dynamics of the conversation or the participants' compatibility]

    5. Suggestion for Next Topic:
    [Based on the conversation so far, suggest a topic or question that could naturally continue the conversation in an engaging way]

    Summary: [Insert your summary here]
    """
        )

        self.chain_suggestion = LLMChain(llm=self.llm, prompt=self.prompt_template_suggestion, memory=self.memory)
        self.chain_score = LLMChain(llm=self.llm, prompt=self.prompt_template_scoring, memory=self.memory)
        self.chain_summarize = LLMChain(llm=self.llm, prompt=self.prompt_template_summarize, memory=self.memory)

    
    def return_suggestion(self):
        chat_history = self.memory.load_memory_variables({})["chat_history"]
        response = self.chain_suggestion.invoke({"human_input   ": "", "chat_history": chat_history})
        suggestion = response['text'].strip()
        return suggestion

    def score_sentence(self, sentence):
        chat_history = self.memory.load_memory_variables({})["chat_history"]
        response = self.chain_score.invoke({"human_input": sentence, "chat_history": chat_history})
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
    
    def summarize_conversation(self):
        chat_history = self.memory.load_memory_variables({})["chat_history"]
        response = self.chain_summarize.invoke({"human_input": "", "chat_history": chat_history})
        print(response)
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