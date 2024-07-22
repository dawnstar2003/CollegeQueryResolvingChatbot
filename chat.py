from database import Database
import random
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize, stem
from autocorrect import Speller
import mysql.connector
import datetime

"""set gpu or cpu (hardware modification)"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Initialize Database instance
db = Database(host="localhost", user="root", password="", database="chatbot_database")
# Fetch data from the database
intents_data = db.fetch_intents_data()

# Process the fetched data and initialize the chatbot
# ...


# Process the fetched data
intents = {'intents': []}
for row in intents_data:
    tag = row[0]
    patterns = row[1].split(';')
    responses = row[2].split(';')
    intents['intents'].append({'tag': tag, 'patterns': patterns, 'responses': responses})
   

# Access the pre-trained model
FILE = "data.pth"
data = torch.load(FILE)

# Initialize the chatbot neural network model
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

"""define the layers of the model"""
model = NeuralNet(input_size, hidden_size, output_size)

"""initializing the model for chatbot"""
model.load_state_dict(model_state)

"""To say its for testing and not training"""
model.eval()

bot_name = "Qbot"

# Track variables for fallback rate accuracy measurement
total_correct_intents = 0

total_queries = 0
fallback_events = 0

prev_input = ''

my_dictionary = {
    'college': ['kcet', 'clg', 'collage', 'kamaraj'],
    'address': ['addr'],
    'contact': ['cnt'],
    'about': ['abt'],
    'yes': ['s', 'yeah', 'hmm'],
    'engineering': ['engg', 'eng'],
    'department': ['dept'],
    'admission': ['admit', 'join', 'enroll'],
    'faculty': ['professors', 'teachers', 'instructors'],
    'courses': ['subjects', 'modules', 'curriculum'],
    'events': ['activities', 'functions', 'programs'],
    'fees': ['tuition', 'costs', 'expenses'],
    'scholarships': ['grants', 'financial aid'],
    'placements': ['jobs', 'career opportunities'],
    'facilities': ['amenities', 'resources'],
    'campus': ['grounds', 'premises'],
    'students': ['pupils', 'learners'],
    'why' : ['y']
}

def accuracty(total_queries, fallback_events, total_correct_intents):
    fallback_rate = (fallback_events / total_queries) * 100
    intent_matching_accuracy = (total_correct_intents / total_queries) * 100
    print(f"Total queries received: {total_queries}")
    print(f"Total fallback events: {fallback_events}")
    print(f"Fallback rate accuracy: {fallback_rate:.2f}%")
    print(f"Total correct intents: {total_correct_intents}")
    print(f"Intent matching accuracy: {intent_matching_accuracy:.2f}%")

#Matching user string to a standard string
def standard_string(dictionary, search_string):
    for key, values in dictionary.items():
        if search_string.lower() in [value.lower() for value in values]:
            return key
    return search_string


# Create a Speller object
spell = Speller(lang='en')

# Function to correct spelling mistakes in a sentence
def correct_spelling(sentence):
    return spell(sentence)

def process_result(input_string):
    result = ""
    for char in input_string:
        if char != '"' and char != '”' and char != '“':
            result += char
    return result
    
# Function to process user input and generate response
def get_response(msg):

    global total_correct_intents,total_queries, fallback_events,prev_input
    total_queries += 1
    msg = msg.lower()
    sentence = tokenize(msg)
    
    if msg != 'yes' and msg != 'no' and msg != 'quit':
        standard_sentence = ""
        for s in sentence:
            standard_sentence += standard_string(my_dictionary, s)
            standard_sentence += " "
            msg = standard_sentence
            userQuery = msg

    msg = msg.lower()
    userQuery = msg
    
    msg = correct_spelling(msg)
    
    print(msg)
    if msg == 'yes':
        print("inside yes")
        msg = prev_input
        userQuery = msg

    elif msg == 'no':
        print("inside no")
        return 'Kindly type your question correcly'

    elif(msg.casefold() != userQuery.casefold()):
        print("inside case forld")
        s = f"'{msg}' is your question ? ... if so type yes"
        prev_input = msg
        return s
    #  print(f"corrected : {msg}")
    
    if msg == 'quit':
        accuracty(total_queries, fallback_events, total_correct_intents)
        return 'Bye'
    
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]


    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]


    # Find the intent corresponding to the predicted tag
    predicted_intent = next((intent for intent in intents['intents'] if intent['tag'] == tag), None)
    
    current_time = datetime.datetime.now()
 

    if predicted_intent:
        # Check if the user query matches any pattern in the predicted intent
        if any(pattern in msg for pattern in predicted_intent['patterns']):
            total_correct_intents += 1
                
    print(prob.item()) 

    if prob.item() > 0.75:
        print("inside prob", prob.item())
        # If the predicted probability is greater than 0.59, check if any intent matches
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(tag, intent["tag"], " tags")
                result = random.choice(intent['responses'])
                print(result,"result")
                if(result == ""):
                    break
                processed_result = process_result(result)
                return processed_result
      
        try:
            insert_query = "INSERT INTO unanswer_question (question) VALUES (%s)"
            
            # Execute the SQL statement with the user's question (stored in msg variable) as the parameter
            db.cursor.execute(insert_query, (msg,))
            
            # Commit the transaction to save the changes to the database
            db.connection.commit()
           
        except Exception as e:
            # Handle any exceptions or errors that occur during the database operation
            print("An error occurred while inserting the question into the database:", e)
        
        # Return the fallback response
        fallback_events += 1
        return f"Sorry, kindly contact our office manager - 94435-44988."
    else:
        # If the predicted probability is not greater than 0.50, increment fallback_events and return the fallback response
        fallback_events += 1
        try:
            insert_query = "INSERT INTO unanswer_question (question) VALUES (%s)"
            
            # Execute the SQL statement with the user's question (stored in msg variable) as the parameter
            db.cursor.execute(insert_query, (msg,))
            
            # Commit the transaction to save the changes to the database
            db.connection.commit()
            
        except Exception as e:
            # Handle any exceptions or errors that occur during the database operation
            print("An error occurred while inserting the question into the database:", e)
       
        return  f"Sorry, Question is not related."

if __name__ == "__main__":

    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)
    accuracty(total_queries, fallback_events, total_correct_intents)
   








 

