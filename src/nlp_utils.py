def words_to_sentence(words):
    """
    Join a list of predicted words into a sentence. Capitalize and add period.
    """
    if not words:
        return ''
    sentence = ' '.join(words)
    # Capitalize first letter and add period
    sentence = sentence[0].upper() + sentence[1:]
    if not sentence.endswith('.'):
        sentence += '.'
    return sentence

# Advanced grammar correction using transformers (T5)
def correct_grammar(sentence):
    try:
        from transformers import pipeline
        # Use a cached pipeline for efficiency
        if not hasattr(correct_grammar, '_pipe'):
            correct_grammar._pipe = pipeline('text2text-generation', model='vennify/t5-base-grammar-correction')
        pipe = correct_grammar._pipe
        result = pipe(sentence, max_length=64)[0]['generated_text']
        return result
    except Exception as e:
        # If transformers not available or error, fallback
        print(f"[NLP] Grammar correction fallback: {e}")
        return sentence 