from transformers import pipeline
# pipeline is a quick API provided by Hugging Face that allows
# you to call common tasks(like question answering) with just a few lines of code.
# without having to write the model loading and inference processes code yourself.

qa_pipeline = pipeline("question-answering") 
# Here I specify the task as "question-answering"
# The pipeline function will automatically download a pre-trained model and tokenizer for this task.

context = "1984 is a novel written by George Orwell in 1949."
question = "Who wrote the novel 1984?"
# QA model needs both a context and a question to provide an answer.

result = qa_pipeline(question=question, context=context)
print(result)

# My code = call the pre-trained QA model → 
# given context + question → the model extracts the answer from the text. 
# So the final answer is: George Orwell.