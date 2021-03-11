def highlight():
    pass
student_answer = None
model_answer = None
THRESHOLD = None

# Iterate through all tokens in the model answer
for model_token in model_answer:
    # Process only tokens that are not in the stop word list and alphanumeric
    if not model_token.is_stop and model_token.is_alpha:
        # Iterate through the tokens in the student's answer
        for answer_token in student_answer:
            # If the the answer token is not a stop word and alphanumeric:
            # Highlight the tokens if their vectors' cosine similarity exceeds the given threshold
            if not answer_token.is_stop and model_token.is_alpha and model_token.similarity(answer_token) > THRESHOLD:
                highlight(model_token)
                highlight(answer_token) 