from chatbot import PublicationQA
import csv
import datetime
from pathlib import Path
from typing import List, Optional
from embedding_bot import QA_bot


def append_to_csv(
    file_path: str,
    data: List[str],
    headers: Optional[List[str]] = None
) -> None:
    """
    Appends data to a CSV file, creating the file if it doesn't exist.

    Args:
        file_path: Path to the CSV file
        data: List of values to append 
        headers: Column headers if creating new file.
    """

    
    file = Path(file_path)
    file_exists = file.is_file()
    has_content = file_exists and file.stat().st_size > 0

    try:
        with open(file_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            if not file_exists or (file_exists and not has_content and headers):
                writer.writerow(headers)
            
            writer.writerow(data)
        
        print(f"Data successfully appended to {file_path}")
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")

def select_model(model_number):
    if model_number==1:
        PUBLICATION_CONTEXT = (
            "https://www.researchgate.net/profile/Gary-Clark/publication/19364043_"
            "Slamon_DJ_Clark_GM_Wong_SG_Levin_WJ_Ullrich_A_McGuire_WLHuman_breast_"
            "cancer_correlation_of_relapse_and_survival_with_amplification_of_the_"
            "HER-2neu_oncogene_Science_Wash_DC_235_177-182/links/0046352b85f241a532"
            "000000/Slamon-DJ-Clark-GM-Wong-SG-Levin-WJ-Ullrich-A-McGuire-WLHuman-"
            "breast-cancer-correlation-of-relapse-and-survival-with-amplification-"
            "of-the-HER-2-neu-oncogene-Science-Wash-DC-235-177-182.pdf"
        )

        qa_bot = PublicationQA(
            model_name="llama3:8b",
            publication_context=PUBLICATION_CONTEXT
        )
    elif model_number==2:
        embedding_type="mxbai-embed-large"
        qa_bot = QA_bot('llama3:8b', embedding_type, embedding_type)
    else:
        embedding_model='nomic-embed-text'
        qa_bot = QA_bot('llama3:8b', embedding_model, embedding_model)
    return qa_bot

        


if __name__ == "__main__":
    """Main execution function for the HER2 QA system."""

    feedback_records = []
    model_input = input(
        "\nType 1: llama 3, type 2 :mxbai, else nomic:"
    ).strip()
    qa_bot=select_model(model_input)

    while True:

        user_input = input(
            "\nType your question about HER2 (or 'exit' to quit): "
        ).strip()
        
        if user_input=='exit':
            
            score = input("On a scale from 0 to 9, how satisfied are you with the output?: ").strip()
            feedback = input( "Any other feedbacks? " ).strip()

            append_to_csv('../doc/feedbacks.csv', [datetime.datetime.now(),score,feedback],['Time','satisfy_score','feedback'])
            break
        answer = qa_bot.ask_question(user_input)
        print(f"\nAnswer: {answer}")
        user_judgement = input(
            "\nDo you think the result is correct(1 for yes, -1 for no, 0 for don't know'): "
        ).strip()

        append_to_csv('../doc/logging.csv', [user_input,answer,user_judgement],['question','answer','user_judgement'])
 