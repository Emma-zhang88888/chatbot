# -*- coding: utf-8 -*-
# create embeddings for document
import pdfplumber
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings import OllamaEmbeddings  # Only works with newer versions

class PDFEmbedder:
    def __init__(self, embedding_model_name):
        """
        Initialize PDF embedder with specified embedding model.
        
        Args:
            embedding_model_name (str): Name of HuggingFace embedding model
        """
        self.embeddings_model = OllamaEmbeddings(model=embedding_model_name)
        self.vectorstore = None

    def extract_text_with_pdfplumber(self, pdf_path):
        """
        Extract text from PDF using pdfplumber.
        
        Args:
            pdf_path (str): Path to PDF file
            
        Returns:
            str: Extracted text
        """
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text(x_tolerance=1, y_tolerance=1) + "\n"
        return text

    def advanced_text_preprocessing(self, text):
        """
        Advanced text preprocessing with tokenization and lemmatization.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned and processed text
        """
        # Basic cleaning
        #text = re.sub(r'\d+', '', text)  # Remove numbers
        #text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        
        # Tokenization
        #tokens = word_tokenize(text.lower())
        return text
        
        # Lemmatization and filtering
        #processed_tokens = []
        #for token in tokens:
        #    if token not in self.stop_words and token not in self.punctuation:
        #        lemma = self.lemmatizer.lemmatize(token)
        #        processed_tokens.append(lemma)
        
        #return ' '.join(processed_tokens)

    def pdf_to_embedding(self,pdf_path,output_name, chunk_size=500, chunk_overlap=100):
        """
        Process PDF file and create embeddings for its content.
        
        Args:
            pdf_path (str): Path to PDF file
            chunk_size (int): Size of text chunks
            chunk_overlap (int): Overlap between chunks
            
        Returns:
            bool: True if embeddings were successfully created and saved
        """
        # Extract and preprocess text
        raw_text = self.extract_text_with_pdfplumber(pdf_path)
        cleaned_text = self.advanced_text_preprocessing(raw_text)
        
        # Split text into documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
        splits = text_splitter.split_text(cleaned_text)
        documents = [Document(page_content=chunk) for chunk in splits]
        # Create and save embeddings
        self.vectorstore = FAISS.from_documents(
            documents=documents, 
            embedding=self.embeddings_model
        )
        
        # Save embeddings
        self.vectorstore.save_local(output_name)
        print("Embeddings created and saved successfully")
        return True

# Example usage
if __name__ == "__main__":
    #"mxbai-embed-large"  'nomic-embed-text'
    embedding_model="mxbai-embed-large"
    embedder = PDFEmbedder(embedding_model)
    pdf_path = "../doc/her2.pdf"
    embedder.pdf_to_embedding(pdf_path,embedding_model)