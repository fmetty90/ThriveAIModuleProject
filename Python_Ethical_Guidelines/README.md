## ethical guideline class code generator
Returns a tuple containing a random ethical guideline number (1-7) 
and the corresponding response message.

'''
import ethical_guidelines as eg

# Example Usage:
user_question = "How many children died on the Titanic?"

guideline_number, response = eg.get_ethical_guideline(user_question)

print(f"Guideline Number: {guideline_number}")
print(f"Response: {response}")    