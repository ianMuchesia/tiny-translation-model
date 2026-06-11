import os
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from src.inference import Translator 

def evaluate_model(translator, test_dataset):
    strong_translations = []
    weak_translations = []
    short_bias_flags = []
    total_bleu = 0.0
    
    # 1. Initialize the NLTK smoothing function to fix the 0.00 bug on short sentences
    smoother = SmoothingFunction().method1

    for eng_text, ref_swahili_raw in test_dataset:
        # 2. Get the prediction
        pred_text = translator.translate(eng_text.strip())
        
        pred_tokens = pred_text.strip().split()
        
        # 3. Split the reference string into a list of words
        ref_tokens = ref_swahili_raw.strip().split()
        
        # 4. Calculate NLTK BLEU WITH the smoothing function applied
        score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoother)
        total_bleu += score
        
        # 5. SWE Logic: Categorize the results
        result_string = f"{eng_text} -> {' '.join(pred_tokens)} (Score: {score:.2f})"
        
        if score > 0.5:
            strong_translations.append(result_string)
        elif score < 0.2:
            weak_translations.append(result_string)
            
        if len(pred_tokens) < (len(ref_tokens) / 2):
            short_bias_flags.append(result_string)

    # 6. Write out the Report
    avg_bleu = total_bleu / len(test_dataset)
    
    os.makedirs("experiments", exist_ok=True)
    with open("./../experiments/bleu_scores.txt", "w") as f:
        f.write(f"Average BLEU: {avg_bleu:.2f}\n\n")
        f.write("Strong:\n" + "\n".join(strong_translations) + "\n\n")
        f.write("Weak:\n" + "\n".join(weak_translations) + "\n\n")
        f.write("Short Sentence Bias Flags:\n" + "\n".join(short_bias_flags))
        
    print("Report generated at experiments/bleu_scores.txt")