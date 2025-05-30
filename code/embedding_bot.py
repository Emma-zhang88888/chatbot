from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
import PDFEmbedder,time

class QA_bot():
    def __init__(self, model, embedding_model, embedding_type):
        """
        Initialize the QA bot with specified models and PDF path.
        
        Args:
            model (str): Name of the Ollama LLM model to use (e.g., 'llama3')
            embedding_model (str): Name of the Ollama embedding model to use
            pdf_path (str): Path to the PDF file to process
        """
        self.llm = Ollama(model=model)
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.vectorstore = None
        self.embedding_type = embedding_type
    
    def load_embeddings(self):
        """
        Load pre-generated embeddings from local storage.
        
        Returns:
            bool: True if embeddings were successfully loaded, False otherwise
            
        Summary:
            Attempts to load embeddings from 'pdf_embeddings' directory.
            Updates the class's vectorstore if successful.
        """
        #print('Loading embeddings...')
        try:
            self.vectorstore = FAISS.load_local(
                self.embedding_type,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            return True
        except Exception as e:
            print(f"Failed to load embeddings: {e}")
            return False

    def ask_question(self, question):
        """
        Query the document embeddings with a question.
        
        Args:
            question (str): The question to ask about the document content
            
        Returns:
            str: The generated answer to the question
            
        Raises:
            ValueError: If no embeddings are loaded
            
        Summary:
            Uses the retrieval-augmented QA chain to find relevant context
            from the embeddings and generate an answer using the LLM.
        """
        if not self.vectorstore:
            if not self.load_embeddings():
                raise ValueError("No embeddings loaded. Run pdf_to_embedding() first.")
        
        qa_chain = RetrievalQA.from_chain_type(
            self.llm,
            retriever=self.vectorstore.as_retriever(),
            chain_type="stuff"
        )
        result = qa_chain.invoke({"query": question})
        return result["result"]

if __name__ == "__main__":
    pdf_path = "../her2.pdf"
    
    #embedding_model="mxbai-embed-large"
    embedding_model='nomic-embed-text'
    
    bot = QA_bot('llama3:8b', embedding_model, embedding_model)
    
    # Create embeddings if they don't exist
    try:
        if not bot.load_embeddings():
            embedder = PDFEmbedder()
            embedder.pdf_to_embedding(pdf_path)
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        embedder = PDFEmbedder()
        embedder.pdf_to_embedding(pdf_path)
    start_time = time.time()

    # Ask a question
    question = 'was rearrangement of the HER-2/neu gene rare? answer yes or no'
    answer = bot.ask_question(question)
    response_time = time.time() - start_time
    print ('response_time',response_time)
    print(answer)