import ollama,time

class PublicationQA:
    def __init__(self, model_name: str, publication_context: str = ""):
        """
        Initialize the Q/A chatbot with a specific LLM model and publication context.
        
        Args:
            model_name: Name of the OLLAMA model to use (default: "mistral")
            publication_context: Background information about the publication
        """
        self.model_name = model_name
        self.publication_context = publication_context
        
        
    def generate_prompt(self, question: str) -> str:
        """Generate a prompt combining the publication context and question."""
        return f"""
        
        Question: {question}
        Find answer from the following text: 
        {self.publication_context}
        
        """
    
    def ask_question(self, question: str, verbose: bool = False)  :
        """Ask a question about the publication and get a response."""
        prompt = self.generate_prompt(question)
        if verbose:
            print(f"Generated prompt:\n{prompt}")
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = ollama.chat(
                    model=self.model_name ,
                    messages=messages
                    )
            answer=response["message"]["content"]
            return answer
            
        except:
            return None
            




if __name__ == "__main__":
    
    pdf_path = "../doc/her2.pdf"
    #PUBLICATION_CONTEXT=extract_text_pypdf2(pdf_path)
    
    #print ('finsih reading doc', len(PUBLICATION_CONTEXT))
    start_time = time.time()
    PUBLICATION_CONTEXT='https://www.researchgate.net/profile/Gary-Clark/publication/19364043_Slamon_DJ_Clark_GM_Wong_SG_Levin_WJ_Ullrich_A_McGuire_WLHuman_breast_cancer_correlation_of_relapse_and_survival_with_amplification_of_the_HER-2neu_oncogene_Science_Wash_DC_235_177-182/links/0046352b85f241a532000000/Slamon-DJ-Clark-GM-Wong-SG-Levin-WJ-Ullrich-A-McGuire-WLHuman-breast-cancer-correlation-of-relapse-and-survival-with-amplification-of-the-HER-2-neu-oncogene-Science-Wash-DC-235-177-182.pdf'
    # Initialize the QA system
    #PUBLICATION_CONTEXT='HER-2/neu was found to be amplified from 2- to greater than 20-fold in 30% of the tumors.'
    qa_bot = PublicationQA(model_name="llama3:8b", publication_context=PUBLICATION_CONTEXT)

        

    # Example questions
    questions = [
        'How many breast malignancy tissues were evaluated in the study of the paper?',
        "Which cancer was discussed in this paper?	"

    ]
    
    # Ask questions and print answers
    for question in questions:
        result = qa_bot.ask_question(question)

    response_time = time.time() - start_time
    print ('response_time',response_time)
