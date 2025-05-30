import csv
import time
from chatbot import PublicationQA
from embedding_bot import QA_bot


def evaluate_chatbot(csv_file, publication_context):
    """Evaluate chatbot performance on questions from a CSV file.
    
    Args:
        csv_file: Path to CSV file containing questions
        publication_context: URL or path to publication for context
        
    Returns:
        List of dictionaries containing question, response, and timing data
    """
    results = []
    
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        for i, row in enumerate(reader, 1):
            if i > 12:  # Limit to first 12 questions for testing
                break
                
            question = row['Question']
            print(f"\n################ {question} ################")
            
            # Measure response time
            start_time = time.time()
            '''
            #### basic CHAT BOT
            qa_bot = PublicationQA(
                model_name="llama3:8b",
                publication_context=publication_context
            )
            '''
            #### EMBEDDING BOT
            embedding_model='mxbai-embed-large'
            qa_bot = QA_bot('llama3:8b', embedding_model, embedding_model)
            
            bot_answer = qa_bot.ask_question(question)
            
            response_time = time.time() - start_time
            print (bot_answer) # embedding_bot
            
            results.append({
                'question': question,
                'response': bot_answer,
                'response_time': response_time,
            })
    
    return results


def print_metrics(results):
    """Print evaluation metrics from the test results.
    
    Args:
        results: List of test results from evaluate_chatbot()
    """
    total_questions = len(results)
    total_time = sum(result['response_time'] for result in results)
    avg_time = total_time / total_questions if total_questions else 0
    
    print("\nEvaluation Metrics:")
    print(f"  Average Response Time: {avg_time:.4f} seconds")
    print(f"  Total Questions Processed: {total_questions}")


if __name__ == "__main__":
    PUBLICATION_CONTEXT = (
        'https://www.researchgate.net/profile/Gary-Clark/publication/'
        '19364043_Slamon_DJ_Clark_GM_Wong_SG_Levin_WJ_Ullrich_A_McGuire_WLHuman_'
        'breast_cancer_correlation_of_relapse_and_survival_with_amplification_of_'
        'the_HER-2neu_oncogene_Science_Wash_DC_235_177-182/links/'
        '0046352b85f241a532000000/Slamon-DJ-Clark-GM-Wong-SG-Levin-WJ-Ullrich-A-'
        'McGuire-WLHuman-breast-cancer-correlation-of-relapse-and-survival-with-'
        'amplification-of-the-HER-2-neu-oncogene-Science-Wash-DC-235-177-182.pdf'
    )
    
    test_results = evaluate_chatbot(
        csv_file='../doc/test_questions.csv',
        publication_context=PUBLICATION_CONTEXT
    )
    
    print_metrics(test_results)

