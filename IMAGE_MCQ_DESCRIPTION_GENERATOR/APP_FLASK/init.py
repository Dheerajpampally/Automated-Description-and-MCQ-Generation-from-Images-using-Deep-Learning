from flask import Flask, render_template, request, jsonify
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
from tensorflow.keras.models import load_model
from pickle import load
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from random import shuffle
from werkzeug.utils import secure_filename
import os
# import gensim
# from fasttext import load_model
from gensim.models import KeyedVectors

UPLOAD_FOLDER = 'D:/MAIN_PRO/APP/uploads'  # Set your desired upload folder path
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}  # Set allowed file extensions

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Load tokenizer and model
tokenizer = load(open("D:/MAIN_PRO/Token/tokenizer.p", "rb"))
model = load_model("D:/MAIN_PRO/model/model_10.h5 ")
xception_model = Xception(include_top=False, pooling="avg")
max_length = 32
word_vectors = KeyedVectors.load("D:/MAIN_PRO/APP/Fast/fasttext_modelfasttext-wiki-news-subwords-300")


# Route for home page
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/generate', methods=['POST'])
def generate_caption():
    if 'file' not in request.files or request.files['file'].filename == '':
        return render_template('ind.html', message='No file uploaded')
    
    file = request.files['file']
    
    # Check if the file is one of the allowed types/extensions
    if file and allowed_file(file.filename):
        # Make the filename safe, remove unsupported chars
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
         
        # Read image file and convert to PIL image
        image = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # Resize image
        image = image.resize((299, 299))
        image = np.array(image)
        # Convert the image from 4 channels to 3 channels
        if image.shape[2] == 4:
            image = image[..., :3]
        image = np.expand_dims(image, axis=0)
        image = image / 255.0  # Normalize pixel values to [0, 1]
        image = image - 1.0
        # Extract features from image using Xception model
        photo = xception_model.predict(image)
        # Generate image caption
        description = generate_desc(model, tokenizer, photo, max_length)
        # Remove temporary image file
        # os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # Return generated caption
        return jsonify(description=description)
    else:
        return 'No image file received'

    
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
           
@app.route('/process', methods=['POST'])
def generate_MCQ_():
    if 'file' not in request.files or request.files['file'].filename == '':
        return render_template('mcq.html', message='No file uploaded')
    
    file = request.files['file']
    
    # Check if the file is one of the allowed types/extensions
    if file and allowed_file(file.filename):
    # if request.method == 'POST' :
    #     file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        image = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        image = image.resize((299, 299))
        image = np.array(image)
            # Convert the image from 4 channels to 3 channels
        if image.shape[2] == 4:
                image = image[..., :3]
        image = np.expand_dims(image, axis=0)
        image = image / 255.0  # Normalize pixel values to [0, 1]
        image = image - 1.0
            # Extract features from image using Xception model
        photo = xception_model.predict(image)
        # if photo is None:
        #     return jsonify({'error': 'Failed to extract features from image'})

        # Generate description for the input image
        description = generate_desc(model, tokenizer, photo, max_length)

        question, options, key_phrases = generate_mcq(description)


        return render_template('mcq.html', question=question, options=options, answer=key_phrases[0], description=description)
    else :
        return render_template('mcq.html', message='Invalid file type')


def generate_mcq(description):
    # Extract key phrases from the description
    stop_words = set(stopwords.words("english"))
    words = wordpunct_tokenize(description)
    words = [word.lower() for word in words if word.isalpha() and word not in stop_words]
    pos_tags = pos_tag(words)
    key_phrases = [word for word, pos in pos_tags if pos in ['NN', 'NNS', 'VBG', 'VBZ', 'VBN', 'VBP']]
    shuffle(key_phrases)
    
    # Generate similar options from the key phrases
    options = generate_option(key_phrases[0])
    options.append(key_phrases[0])
    shuffle(options)
    
    # Create a fill-in-the-blanks question with options
    question = description.replace(key_phrases[0], '_____')
    
    return question, options, key_phrases

def generate_option(answer):
    answer=answer.split(" ")
    options=[]
    if(len(answer)>1):
        similar_words = word_vectors.most_similar(answer[0], topn=20)
        similar = [w[0] for w in similar_words if w[0][0].lower() !=answer[0][0]]
        for option in similar:
            if len(answer)==2:
                options.append(option+" "+answer[1])
            else:
                options.append(option+" "+answer[1]+" "+answer[2])
      
    else:
        similar_words = word_vectors.most_similar(answer[0], topn=20)
        similar = [w[0] for w in similar_words if w[0][0].lower() !=answer[0][0]]
        for option in similar:
            options.append(option)
    return options[:4] # Get the first four similar options


# Utility function to generate image caption
def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text.replace('start', '').replace('end', '')

# Utility function to get word for given index
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer and word != 'start' and word != 'end':
            return word
    return None

if __name__ == '__main__':
    app.run(debug=True)