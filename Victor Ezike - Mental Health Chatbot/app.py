# from flask import Flask, render_template, request
# import json
# import random
# import nltk
# import pickle
# import numpy as np
# from keras.models import load_model
# from nltk.stem import WordNetLemmatizer

# nltk.download('popular')
# nltk.download('punkt')


# # Initialize the lemmatizer
# lemmatizer = WordNetLemmatizer()

# # Load the pre-trained model, intents, words, and classes
# model = load_model('model.h5')
# intents = json.loads(open('intents.json').read())
# words = pickle.load(open('texts.pkl', 'rb'))
# classes = pickle.load(open('labels.pkl', 'rb'))

# # Function to clean up the sentence
# def clean_up_sentence(sentence):
#     sentence_words = nltk.word_tokenize(sentence)
#     sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
#     return sentence_words

# # Function to create a bag of words
# def bow(sentence, words, show_details=True):
#     sentence_words = clean_up_sentence(sentence)
#     bag = [0] * len(words)
#     for s in sentence_words:
#         for i, w in enumerate(words):
#             if w == s:
#                 bag[i] = 1
#                 if show_details:
#                     print("found in bag: %s" % w)
#     return np.array(bag)

# # Function to predict the class of the sentence
# def predict_class(sentence, model):
#     p = bow(sentence, words, show_details=False)
#     res = model.predict(np.array([p]))[0]
#     ERROR_THRESHOLD = 0.25
#     results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
#     results.sort(key=lambda x: x[1], reverse=True)
#     return_list = []
#     for r in results:
#         return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
#     return return_list

# # Function to get the response based on the predicted class
# def getResponse(ints, intents_json):
#     if ints:
#         tag = ints[0]['intent']
#         list_of_intents = intents_json['intents']
#         for i in list_of_intents:
#             if i['tag'] == tag:
#                 result = random.choice(i['responses'])
#                 break
#         return result
#     else:
#         return "Sorry, I didn't understand that."

# # Function to generate a chatbot response
# def chatbot_response(msg):
#     res = getResponse(predict_class(msg, model), intents)
#     return res

# # Initialize the Flask app
# app = Flask(__name__)
# app.static_folder = 'static'

# # Route for the homepage
# @app.route("/")
# def home():
#     return render_template("index.html")

# # Route to get the bot response
# @app.route("/get")
# def get_bot_response():
#     userText = request.args.get('msg')
#     print("get_bot_response:- " + userText)
#     chatbot_response_text = chatbot_response(userText)
#     return chatbot_response_text

# # Run the app
# if __name__ == "__main__":
#     app.run(debug=True)



from flask import Flask, render_template, request
import json
import random
import nltk
import pickle
import numpy as np
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import ssl

# Fix for SSL certificate verification
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the pre-trained model, intents, words, and classes
model = load_model('model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))

# Function to clean up the sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Function to create a bag of words
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

# Function to predict the class of the sentence
def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# Function to get the response based on the predicted class
def getResponse(ints, intents_json):
    if ints:
        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
        return result
    else:
        return "Sorry, I didn't understand that."

# Function to generate a chatbot response
def chatbot_response(msg):
    res = getResponse(predict_class(msg, model), intents)
    return res

# Initialize the Flask app
app = Flask(__name__)
app.static_folder = 'static'

# Route for the homepage
@app.route("/")
def home():
    return render_template("index.html")

# Route to get the bot response
@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    print("get_bot_response:- " + userText)
    chatbot_response_text = chatbot_response(userText)
    return chatbot_response_text

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
