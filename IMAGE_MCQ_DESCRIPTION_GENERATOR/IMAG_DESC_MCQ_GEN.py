#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !pip install tensorflow keras pillow numpy tqdm


# In[4]:


pip install keras --upgrade


# In[5]:


import string
import numpy as np
from PIL import Image
import os
from pickle import dump, load
import numpy as np


# In[63]:


pip install pydot


# In[6]:


from keras.applications.xception import Xception, preprocess_input
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.text import Tokenizer
from tqdm import tqdm_notebook as tqdm
tqdm().pandas()


# In[7]:


# from keras.layers.merge import add
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, LSTM, concatenate, add
from keras.models import Model, load_model
from sklearn.metrics import classification_report, accuracy_score


# In[8]:


from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
# from keras.utils import to_categorical
from keras.layers import add
from keras.models import Model, load_model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout


# In[9]:


# Loading a text file into memory
def load_doc(filename):
    file=open(filename,'r')
    text=file.read()
    file.close()
    return text


# In[10]:


# get all imgs with their captions
def img_cap(filename):
    file=load_doc(filename)
    captions1=file.split('\n')
    descriptions1={}
    for caption in captions1[:-1]:
        img,caption = caption.split('\t')
        if img[:-2] not in descriptions1:
            descriptions1[img[:-2]]=[caption]
        else:
            descriptions1[img[:-2]].append(caption)
    return descriptions1
            


# In[11]:


#Data cleaning- lower casing, removing puntuations and words containing numbers
def cleaning(captions1):
    table=str.maketrans(" "," ",string.punctuation)
    for img,caps in captions1.items():
        for i,imgcap in enumerate(caps):
            
            imgcap.replace("-"," ")
            desc=imgcap.split()
            
            desc= [word.lower() for word in desc]
            desc=[word.translate(table) for word in desc]
            desc=[word for word in desc if(len(word)>1)]
            desc=[word for word in desc if(word.isalpha())]
            
            imgcap=' '.join(desc)
            captions1[img][i]=imgcap
    return captions1


# In[12]:


# build vocabulary of all unique words
def textvoc(descriptions1):
    vocab=set()
    for key in descriptions1.keys():
        [vocab.update(d.split()) for d in descriptions1[key]]
    return vocab
    


# In[13]:


#All descriptions in one file 
def save_desc(descriptions1,filename):
    lines=[]
    for key,desc_list in descriptions1.items():
        for desc in desc_list:
            lines.append(key+ '\t' +desc)
    data="\n".join(lines)
    file=open(filename,'w')
    file.write(data)
    file.close()
            
            


# In[14]:


# load image path 
dataset_img="C:/Users/dheer/OneDrive/Desktop/practice_work/Flicker8k_Dataset"


# In[20]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# Set the path to the directory containing the images
image_dir = dataset_img

# Loop through all the image files in the directory
for filename in os.listdir(image_dir):
    # Read the image file
    img = mpimg.imread(os.path.join(image_dir, filename))
    img1=image_dir.resize(dataset_img)
    
    # Display the image
    plt.imshow(img)
    plt.axis('off')
    plt.show()


# In[2]:


# load text file path
dataset_text="C:/Users/dheer/OneDrive/Desktop/practice_work/Flickr8k_text"


# In[22]:


# load the text data to prepeocess
filename=dataset_text+ "/" +"Flickr8k.token.txt"


# In[23]:


#mapping them into descriptions dictionary img to 5 captions
descriptions1=img_cap(filename)


# In[24]:


print(descriptions1)


# In[25]:


print("Length of description:", len(descriptions1))


# In[26]:


#cleaning the descriptions
clean_descrip=cleaning(descriptions1)


# In[27]:


print(clean_descrip)


# In[28]:


#building vocabulary 
vocabulary=textvoc(clean_descrip)


# In[29]:


print(vocabulary)


# In[30]:


print("Length of vocabulary:", len(vocabulary))


# In[31]:


#saving each description to file 
save_description=save_desc(clean_descrip, "C:/Users/dheer/OneDrive/Desktop/practice_work/descriptions1.txt")


# In[32]:


print(clean_descrip)


# In[39]:


# extracting the features from all images using xception 
def extract_features(directory):
        model = Xception( include_top=False, pooling='avg' )
        features = {}
        for img in tqdm(os.listdir(directory)):
            filename = directory + "/" + img
            image = Image.open(filename)
            image = image.resize((299,299))
            image = np.expand_dims(image, axis=0)
            image = image/127.5
            image = image - 1.0

            feature = model.predict(image)
            features[img] = feature
        return features

#2048 feature vector
features = extract_features(dataset_img)
dump(features, open("C:/Users/dheer/OneDrive/Desktop/practice_work/features.p","wb"))


# In[30]:


features = load(open("C:/Users/dheer/OneDrive/Desktop/practice_work/features.p","rb"))


# In[31]:


print(features)


# In[12]:


import pickle

# open the pickle file in binary mode
with open("C:/Users/dheer/OneDrive/Desktop/practice_work/features.p", "rb") as f:
    # load the contents of the pickle file into a variable
    data = pickle.load(f)


# In[32]:


print(data)


# In[15]:


#load the data 
def load_photos(filename):
    file = load_doc(filename)
    photos = file.split("\n")[:-1]
    return photos


def load_clean_descriptions(filename, photos): 
    #loading clean_descriptions
    file = load_doc(filename)
    descriptions = {}
    for line in file.split("\n"):

        words = line.split()
        if len(words)<1 :
            continue

        image, image_caption = words[0], words[1:]

        if image in photos:
            if image not in descriptions:
                descriptions[image] = []
            desc = '<start> ' + " ".join(image_caption) + ' <end>'
            descriptions[image].append(desc)

    return descriptions


def load_features(photos):
    #loading all features
    all_features = load(open("C:/Users/dheer/OneDrive/Desktop/practice_work/features.p","rb"))
    #selecting only needed features
    features = {k:all_features[k] for k in photos}
    return features


filename = dataset_text + "/" + "Flickr_8k.trainImages.txt"

#train = loading_data(filename)
train_imgs = load_photos(filename)
train_descriptions = load_clean_descriptions("C:/Users/dheer/OneDrive/Desktop/practice_work/descriptions1.txt", train_imgs)
train_features = load_features(train_imgs)


filename2 = dataset_text + "/" + "Flickr_8k.testImages.txt"

#train = loading_data(filename)
val_imgs = load_photos(filename)
val_descriptions = load_clean_descriptions("C:/Users/dheer/OneDrive/Desktop/practice_work/descriptions1.txt", val_imgs)
val_features = load_features(val_imgs)


# In[26]:


print(train_imgs)


# In[34]:


print(train_descriptions)


# In[35]:


print(train_features)


# In[36]:


#converting dictionary to clean list of descriptions
def dict_to_list(descriptions):
    all_desc = []
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

#creating tokenizer class 
#this will vectorise text corpus
#each integer will represent token in dictionary

from keras.preprocessing.text import Tokenizer

def create_tokenizer(descriptions):
    desc_list = dict_to_list(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(desc_list)
    return tokenizer

# give each word an index, and store that into tokenizer.p pickle file
tokenizer = create_tokenizer(train_descriptions)
dump(tokenizer, open('C:/Users/dheer/OneDrive/Desktop/practice_work/tokenizer.p', 'wb'))
vocab_size = len(tokenizer.word_index) + 1
vocab_size


# In[37]:


Token=load(open("C:/Users/dheer/OneDrive/Desktop/practice_work/tokenizer.p","rb"))


# In[38]:


print(Token)


# In[39]:


#calculate maximum length of descriptions
def max_length(descriptions):
    desc_list = dict_to_list(descriptions1)
    return max(len(d.split()) for d in desc_list)
    
max_length = max_length(descriptions1)
max_length


# In[40]:


#create input-output sequence pairs from the image description.

#data generator, used by model.fit_generator()
def data_generator(descriptions, features, tokenizer, max_length):
    while 1:
        for key, description_list in descriptions.items():
            #retrieve photo features
            feature = features[key][0]
            input_image, input_sequence, output_word = create_sequences(tokenizer, max_length, description_list, feature)
            yield ([input_image, input_sequence], output_word)

def create_sequences(tokenizer, max_length, desc_list, feature):
    X1, X2, y = list(), list(), list()
    # walk through each description for the image
    for desc in desc_list:
        # encode the sequence
        seq = tokenizer.texts_to_sequences([desc])[0]
        # split one sequence into multiple X,y pairs
        for i in range(1, len(seq)):
            # split into input and output pair
            in_seq, out_seq = seq[:i], seq[i]
            # pad input sequence
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            # encode output sequence
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            # store
            X1.append(feature)
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

#You can check the shape of the input and output for your model
[a,b],c = next(data_generator(train_descriptions, features, tokenizer, max_length))
a.shape, b.shape, c.shape
#((47, 2048), (47, 32), (47, 7577))


# In[44]:


from tensorflow.keras.utils import plot_model

# define the captioning model
def define_model(vocab_size, max_length):

    # features from the CNN model squeezed from 2048 to 256 nodes
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    # LSTM sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    # Merging both models
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # summarize model
    print(model.summary())
    plot_model(model, to_file='C:/Users/dheer/OneDrive/Desktop/practice_work/model.png', show_shapes=True)

    return model


# In[45]:


# train our model
print('Dataset: ', len(train_imgs))
print('Descriptions: train=', len(train_descriptions))
print('Photos: train=', len(train_features))
print('Vocabulary Size:', vocab_size)
print('Description Length: ', max_length)
 
val_generator = data_generator(val_descriptions, val_features, tokenizer, max_length)

# Train our model
model = define_model(vocab_size, max_length)
print(model, 'model')
epochs = 10
steps = len(train_descriptions)
val_steps = len(val_descriptions)

# making a directory models to save our models
os.mkdir("C:/Users/dheer/OneDrive/Desktop/practice_work/modelsss")

for i in range(epochs):
    generator = data_generator(train_descriptions, train_features, tokenizer, max_length)
    model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    # save the model after each epoch
    model.save("C:/Users/dheer/OneDrive/Desktop/practice_work/modelsss/model_" + str(i) + ".h5")

    


# In[ ]:





# In[23]:


get_ipython().system('pip install gradio')


# In[28]:


from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
from keras.models import load_model
from pickle import load
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse

img_path = 'C:/Users/dheer/OneDrive/Desktop/practice_work/cc.jpg'

def extract_features(filename, model):
    try:
        image = Image.open(filename)

    except:
        print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
    # Resize the image
    image = image.resize((299, 299))
    image = np.array(image)
    # Convert the image from 4 channels to 3 channels
    if image.shape[2] == 4: 
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image/127.5
    image = image - 1.0
    feature = xception_model.predict(image)
    return feature

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer and word != 'start' and word != 'end': # skip 'start' and 'end' tokens
            return word
    return None

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
    return in_text.replace('start', '').replace('end', '') # remove 'start' and 'end' tokens from the generated description

max_length = 32
tokenizer = load(open("C:/Users/dheer/OneDrive/Desktop/practice_work/tokenizer.p","rb"))
model = load_model('C:/Users/dheer/OneDrive/Desktop/practice_work/modelsss/model_9.h5')
xception_model = Xception(include_top=False, pooling="avg")

photo = extract_features(img_path, xception_model)
img = Image.open(img_path)

description = generate_desc(model, tokenizer, photo, max_length)
print("\n\n")
print(description)
plt.imshow(img)


# In[10]:


from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
from keras.models import load_model
from pickle import load
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse

# Specify the path to the image you want to generate description for
img_path = 'C:/Users/dheer/OneDrive/Desktop/practice_work/304408047_98bab3ea64.jpg'

def extract_features(filename, model):
    # Load and preprocess the image
    try:
        image = Image.open(filename)
    except:
        print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
        return None
    
    # Resize the image
    image = image.resize((299, 299))
    image = np.array(image)
    
    # Convert the image from 4 channels to 3 channels
    if image.shape[2] == 4: 
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image/127.5
    image = image - 1.0
    
    # Extract features using the Xception model
    feature = xception_model.predict(image)
    return feature

def word_for_id(integer, tokenizer):
    # Convert integer to word using the tokenizer's word index
    for word, index in tokenizer.word_index.items():
        if index == integer and word != 'start' and word != 'end': # skip 'start' and 'end' tokens
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    # Generate description given the photo input
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
    return in_text.replace('start', '').replace('end', '') # remove 'start' and 'end' tokens from the generated description

max_length = 32
tokenizer = load(open("C:/Users/dheer/OneDrive/Desktop/practice_work/tokenizer.p","rb"))
model = load_model('C:/Users/dheer/OneDrive/Desktop/practice_work/modelsss/model_9.h5')
xception_model = Xception(include_top=False, pooling="avg")

photo = extract_features(img_path, xception_model)
if photo is not None:
    description = generate_desc(model, tokenizer, photo, max_length)
    print("\n\n")
    print(description)
    img = Image.open(img_path)
    plt.imshow(img)


# In[7]:


from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize


# In[3]:


import gensim


# In[11]:


import gensim.downloader as api

# Load pre-trained word2vec model
modelz = api.load("word2vec-google-news-300")


# In[76]:


save_dir = "D:/api_model"

# Save the model to the specified directory
modelz.save(save_dir + "word2vec-google-news-300.model")

print("Word2Vec model saved successfully!")


# In[24]:


from transformers import (
    TokenClassificationPipeline,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from transformers.pipelines import AggregationStrategy
import numpy as np

# Define keyphrase extraction pipeline
class KeyphraseExtractionPipeline(TokenClassificationPipeline):
    def __init__(self, model, *args, **kwargs):
        super().__init__(
            model=AutoModelForTokenClassification.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            *args,
            **kwargs
        )

    def postprocess(self, model_outputs):
        results = super().postprocess(
            all_outputs=model_outputs,
            aggregation_strategy=AggregationStrategy.FIRST,
        )
        return np.unique([result.get("word").strip() for result in results])
        # return results


# In[25]:


# Load pipeline
model_name = "ml6team/keyphrase-extraction-distilbert-inspec"
extractor = KeyphraseExtractionPipeline(model=model_name)


# In[27]:


extractor.save_pretrained("C:/Users/dheer/OneDrive/Desktop/practice_work/keyphrase-extraction-distilbert-inspec")


# In[28]:


model2 = KeyphraseExtractionPipeline(model="C:/Users/dheer/OneDrive/Desktop/practice_work/keyphrase-extraction-distilbert-inspec")


# In[37]:


get_ipython().system('pip install rake-nltk')


# In[ ]:





# In[5]:


from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
from tensorflow.keras.models import load_model
from pickle import load
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from random import shuffle

# Specify the path to the image you want to generate description for
img_path = 'C:/Users/dheer/OneDrive/Desktop/practice_work/Nila.jpg'

def extract_features(filename, model):
    # Load and preprocess the image
    try:
        image = Image.open(filename)
    except:
        print("ERROR: Couldn't open image! Make sure the image path and extension are correct")
        return None
    
    # Resize the image
    image = image.resize((299, 299))
    image = np.array(image)
    
    # Convert the image from 4 channels to 3 channels
    if image.shape[2] == 4: 
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image/127.5
    image = image - 1.0
    
    # Extract features using the Xception model
    feature = xception_model.predict(image)
    return feature

def word_for_id(integer, tokenizer):
    # Convert integer to word using the tokenizer's word index
    for word, index in tokenizer.word_index.items():
        if index == integer and word != 'start' and word != 'end': # skip 'start' and 'end' tokens
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    # Generate description given the photo input
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
    return in_text.replace('start', '').replace('end', '') # remove 'start' and 'end' tokens from the generated description

def generate_mcq(description):
    # Extract key phrases from the description
    stop_words = set(stopwords.words("english"))
    words = wordpunct_tokenize(description)
    words = [word.lower() for word in words if word.isalpha() and word not in stop_words]
    pos_tags = pos_tag(words)
    key_phrases = [word for word, pos in pos_tags if pos in ['NN', 'NNS', 'VBG', 'VBZ', 'VBN', 'VBP']]
    shuffle(key_phrases)
    
    # Create a fill-in-the-blanks question with options
    question = description.replace(key_phrases[0], '_____')
    options = key_phrases[1:5]
    options.append(key_phrases[0])
    shuffle(options)
    
    return question, options, key_phrases

max_length = 32
tokenizer = load(open("C:/Users/dheer/OneDrive/Desktop/practice_work/tokenizer.p", "rb"))
model = load_model("C:/Users/dheer/OneDrive/Desktop/practice_work/modelsss/model_9.h5")
xception_model = Xception(include_top=False, pooling="avg")

# Extract features from the input image
photo = extract_features(img_path, xception_model)

# Generate description for the input image
description = generate_desc(model, tokenizer, photo, max_length)

# Generate MCQ from the generated description
question, options, key_phrases = generate_mcq(description)

print("Generated Question:")
print(question)
print("\nOptions:")
for i, option in enumerate(options):
    if option == key_phrases[0]: # Check if the current option is the correct option
        correct_option = chr(65+i) # Get the correct option letter (A, B, C, D, etc.)
    print(f"{chr(65+i)}) {option}")

print("\nCorrect Option:")
print(correct_option)
img = Image.open(img_path)
plt.imshow(img)


# In[3]:


model_name = 'fasttext-wiki-news-subwords-300'
word_vectors = api.load(model_name)


# In[4]:


import gensim.downloader as api
word_vectors = api.load(model_name)


# In[60]:


save_dir = "D:/Fast/fasttext_model"

# Save the model to the specified directory
model.save(save_dir + model_name)


# In[9]:


# Load pipeline
model_name = "ml6team/keyphrase-extraction-distilbert-inspec"
extractor = KeyphraseExtractionPipeline(model=model_name)
extractor.save_pretrained("D:\Key")


# In[10]:


model2 = KeyphraseExtractionPipeline(model="D:\Key")


# In[5]:


from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
from tensorflow.keras.models import load_model
from pickle import load
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from random import shuffle

# Specify the path to the image you want to generate description for
img_path = 'C:/Users/dheer/OneDrive/Desktop/practice_work/304408047_98bab3ea64.jpg'

def extract_features(filename, model):
    # Load and preprocess the image
    try:
        image = Image.open(filename)
    except:
        print("ERROR: Couldn't open image! Make sure the image path and extension are correct")
        return None
    
    # Resize the image
    image = image.resize((299, 299))
    image = np.array(image)
    
    # Convert the image from 4 channels to 3 channels
    if image.shape[2] == 4: 
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image/127.5
    image = image - 1.0
    
    # Extract features using the Xception model
    feature = xception_model.predict(image)
    return feature

def word_for_id(integer, tokenizer):
    # Convert integer to word using the tokenizer's word index
    for word, index in tokenizer.word_index.items():
        if index == integer and word != 'start' and word != 'end': # skip 'start' and 'end' tokens
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    # Generate description given the photo input
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
    return in_text.replace('start', '').replace('end', '') # remove 'start' and 'end' tokens from the generated description

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

# Usage:
max_length = 32
tokenizer = load(open("C:/Users/dheer/OneDrive/Desktop/practice_work/tokenizer.p", "rb"))
model = load_model("C:/Users/dheer/OneDrive/Desktop/practice_work/modelsss/model_9.h5")
xception_model = Xception(include_top=False, pooling="avg")

# Extract features from the input image
photo = extract_features(img_path, xception_model)

# Generate description for the input image
description = generate_desc(model, tokenizer, photo, max_length)

question, options, key_phrases = generate_mcq(description)

# Print the generated question and options
print("Generated Question:")
print(question)
print("\nOptions:")
for i, option in enumerate(options):
    if option == key_phrases[0]: # Check if the current option is the correct option
        correct_option = chr(65+i) # Get the correct option letter (A, B, C, D, etc.)
    print(f"{chr(65+i)}) {option}")
print("Answer: ", key_phrases[0])

# print("\nCorrect Option:")
# print(correct_option)
img = Image.open(img_path)
plt.imshow(img)


# In[ ]:


from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
from tensorflow.keras.models import load_model
from pickle import load
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from random import shuffle

# Specify the path to the image you want to generate description for
img_path = 'C:/Users/dheer/OneDrive/Desktop/practice_work/Nila.jpg'

def extract_features(filename, model):
    # Load and preprocess the image
    try:
        image = Image.open(filename)
    except:
        print("ERROR: Couldn't open image! Make sure the image path and extension are correct")
        return None
    
    # Resize the image
    image = image.resize((299, 299))
    image = np.array(image)
    
    # Convert the image from 4 channels to 3 channels
    if image.shape[2] == 4: 
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image/127.5
    image = image - 1.0
    
    # Extract features using the Xception model
    feature = xception_model.predict(image)
    return feature

def word_for_id(integer, tokenizer):
    # Convert integer to word using the tokenizer's word index
    for word, index in tokenizer.word_index.items():
        if index == integer and word != 'start' and word != 'end': # skip 'start' and 'end' tokens
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    # Generate description given the photo input
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
    return in_text.replace('start', '').replace('end', '') # remove 'start' and 'end' tokens from the generated description

def generate_mcq(description):
    # Extract key phrases from the description
    stop_words = set(stopwords.words("english"))
    words = wordpunct_tokenize(description)
    words = [word.lower() for word in words if word.isalpha() and word not in stop_words]
    pos_tags = pos_tag(words)
    key_phrases = [word for word, pos in pos_tags if pos in ['NN', 'NNS', 'VBG', 'VBZ', 'VBN', 'VBP']]
    shuffle(key_phrases)
    
    # Create a fill-in-the-blanks question with options
    question = description.replace(key_phrases[0], '_____')
    options = key_phrases[1:5]
    options.append(key_phrases[0])
    shuffle(options)
    
    return question, options, key_phrases

max_length = 32
tokenizer = load(open("C:/Users/dheer/OneDrive/Desktop/practice_work/tokenizer.p", "rb"))
model = load_model("C:/Users/dheer/OneDrive/Desktop/practice_work/modelsss/model_9.h5")
xception_model = Xception(include_top=False, pooling="avg")

# Extract features from the input image
photo = extract_features(img_path, xception_model)

# Generate description for the input image
description = generate_desc(model, tokenizer, photo, max_length)

# Generate MCQ from the generated description
question, options, key_phrases = generate_mcq(description)

print("Generated Question:")
print(question)
print("\nOptions:")
for i, option in enumerate(options):
    if option == key_phrases[0]: # Check if the current option is the correct option
        correct_option = chr(65+i) # Get the correct option letter (A, B, C, D, etc.)
    print(f"{chr(65+i)}) {option}")

print("\nCorrect Option:")
print(correct_option)
img = Image.open(img_path)
plt.imshow(img)


# In[93]:


from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
from tensorflow.keras.models import load_model
from pickle import load
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from random import shuffle

# Specify the path to the image you want to generate description for
img_path = 'C:/Users/dheer/OneDrive/Desktop/practice_work/doggy.jpg'

def extract_features(filename, model):
    # Load and preprocess the image
    try:
        image = Image.open(filename)
    except:
        print("ERROR: Couldn't open image! Make sure the image path and extension are correct")
        return None
    
    # Resize the image
    image = image.resize((299, 299))
    image = np.array(image)
    
    # Convert the image from 4 channels to 3 channels
    if image.shape[2] == 4: 
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image/127.5
    image = image - 1.0
    
    # Extract features using the Xception model
    feature = xception_model.predict(image)
    return feature

def word_for_id(integer, tokenizer):
    # Convert integer to word using the tokenizer's word index
    for word, index in tokenizer.word_index.items():
        if index == integer and word != 'start' and word != 'end': # skip 'start' and 'end' tokens
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    # Generate description given the photo input
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
    return in_text.replace('start', '').replace('end', '') # remove 'start' and 'end' tokens from the generated description

def generate_mcq(description):
    # Extract key phrases from the description
    stop_words = set(stopwords.words("english"))
    words = wordpunct_tokenize(description)
    words = [word.lower() for word in words if word.isalpha() and word not in stop_words]
    pos_tags = pos_tag(words)
    key_phrases = [word for word, pos in pos_tags if pos in ['NN', 'NNS', 'VBG', 'VBZ', 'VBN', 'VBP']]
    shuffle(key_phrases)
    
    # Create a fill-in-the-blanks question with options
    question = description.replace(key_phrases[0], '_____')
    options = key_phrases[1:5]
    options.append(key_phrases[0])
    shuffle(options)
    
    return question, options

max_length = 32
tokenizer = load(open("C:/Users/dheer/OneDrive/Desktop/practice_work/tokenizer.p", "rb"), encoding='utf-8')
model = load_model("C:/Users/dheer/OneDrive/Desktop/practice_work/modelsss/model_9.h5")
xception_model = Xception(include_top=False, pooling="avg")
# Specify the path to the image you want to generate description for
img_path = 'C:/Users/dheer/OneDrive/Desktop/practice_work/doggy.jpg'

# Extract features from the image
photo = extract_features(img_path, xception_model)

# Generate description for the image
if photo is not None:
    description = generate_desc(model, tokenizer, photo, max_length)
    print("Generated Description: ", description)
    question, options = generate_mcq(description)
    print("Fill-in-the-Blanks Question: ", question)
    print("Options: ", options)


# In[99]:


from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
from tensorflow.keras.models import load_model
from pickle import load
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from random import shuffle
from sklearn.metrics.pairwise import cosine_similarity

# Specify the path to the image you want to generate description for
img_path = 'C:/Users/dheer/OneDrive/Desktop/practice_work/doggy.jpg'

def extract_features(filename, model):
    # Load and preprocess the image
    try:
        image = Image.open(filename)
    except:
        print("ERROR: Couldn't open image! Make sure the image path and extension are correct")
        return None
    
    # Resize the image
    image = image.resize((299, 299))
    image = np.array(image)
    
    # Convert the image from 4 channels to 3 channels
    if image.shape[2] == 4: 
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image/127.5
    image = image - 1.0
    
    # Extract features using the Xception model
    feature = xception_model.predict(image)
    return feature

def word_for_id(integer, tokenizer):
    # Convert integer to word using the tokenizer's word index
    for word, index in tokenizer.word_index.items():
        if index == integer and word != 'start' and word != 'end': # skip 'start' and 'end' tokens
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    # Generate description given the photo input
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
    return in_text.replace('start', '').replace('end', '') # remove 'start' and 'end' tokens from the generated description

def generate_mcq(description):
    # Extract key phrases from the description
    stop_words = set(stopwords.words("english"))
    words = wordpunct_tokenize(description)
    words = [word.lower() for word in words if word.isalpha() and word not in stop_words]
    pos_tags = pos_tag(words)
    key_phrases = [word for word, pos in pos_tags if pos in ['NN', 'NNS', 'VBG', 'VBZ', 'VBN', 'VBP']]
    shuffle(key_phrases)
    
    # Create a fill-in-the-blanks question with options
    question = description.replace(key_phrases[0], '_____')
    options = key_phrases[1:5]
    options.append(key_phrases[0])
    shuffle(options)
    
    return question, options

def get_similar_options(correct_answer, options):
    # Use cosine similarity to get options similar to the correct answer
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([correct_answer] + options)
    similarity_scores = cosine_similarity(vectors[0], vectors[1:])
    sorted_indices = np.argsort(similarity_scores, axis=1)[:,::-1]
    sorted_options = [options[idx] for idx in sorted_indices]
    
    # Display similar options to the correct answer
    print("Similar options to the correct answer:")
    for option in sorted_options:
        print(option)
    
    # Return the sorted options
    return sorted_options
max_length = 32

# Load the tokenizer
tokenizer = load(open("C:/Users/dheer/OneDrive/Desktop/practice_work/tokenizer.p", "rb"))

# Load the model
model = load_model("C:/Users/dheer/OneDrive/Desktop/practice_work/modelsss/model_9.h5")

# Load the Xception model
xception_model = Xception(include_top=False, pooling="avg")

# Extract features from the input image
photo = extract_features(img_path, xception_model)

# Generate description for the input image
description = generate_desc(model, tokenizer, photo, max_length)

# Generate MCQ from the generated description
question, options = generate_mcq(description)

# Add correct answer to the options
# correct_answer = word_for_id(tokenizer.word_index[description.split()[0]], tokenizer)
# options.append(correct_answer)

# Print the generated question and options
print("Generated Question:")
print(question)
print("\nGenerated Options:")
for i, option in enumerate(options):
    print(f"{chr(65+i)}) {option}")


# In[3]:


pip install torch


# In[16]:


# my_list = [1, 2, 3, 4, 5]
# # result = "1, 2, 3, 4, 5"  # Replace ellipsis with a valid string or sequence
# split_result = result.split(", ")  # Call split() on a valid string or sequence


# In[27]:


from nltk.translate.bleu_score import sentence_bleu

img_path = 'C:/Users/dheer/OneDrive/Desktop/practice_work/bike.jpg'

# Load the tokenizer and model
max_length = 32
tokenizer = load(open("C:/Users/dheer/OneDrive/Desktop/practice_work/tokenizer.p","rb"))
model = load_model('C:/Users/dheer/OneDrive/Desktop/practice_work/modelsss/model_9.h5')
xception_model = Xception(include_top=False, pooling="avg")

# Extract features from the input image
photo = extract_features(img_path, xception_model)
img = Image.open(img_path)

# Generate the caption
generated_caption = generate_desc(model, tokenizer, photo, max_length)

# Convert the generated caption to a list of words
generated_caption = generated_caption.split()

# Use the generated caption as the reference caption
reference_captions = [generated_caption]

# Calculate the BLEU score
bleu_score = sentence_bleu(reference_captions, generated_caption)
print("BLEU Score: ", bleu_score)


# In[11]:


from nltk.translate.meteor_score import meteor_score


# In[18]:


reference_path = 'C:/Users/dheer/OneDrive/Desktop/practice_work/descriptions1.txt'

with open(reference_path, 'r') as file:
    data = file.read()

# Split the file contents into image id-description pairs
image_pairs = data.strip().split('\n\n')

# Parse each image id-description pair into separate variables
reference_images = []
reference_descriptions = []
for pair in image_pairs:
    image_id, *descriptions = pair.split('\n')
    reference_images.append(image_id.strip())
    reference_descriptions.append(descriptions)

# Convert the reference descriptions to a list of lists of words
reference_sentences = []
for description_list in reference_descriptions:
    sentences = []
    for description in description_list:
        sentences.append(description.split())
    reference_sentences.append(sentences)

# Compute the METEOR score
# Convert the generated description to a list of words
    generated_sentence = description.split()

# Compute the METEOR score for each reference sentence and take the average
    meteor_scores = []
for reference in reference_sentences:
    score = meteor_score(reference, generated_sentence)
    meteor_scores.append(score)
meteor_score_avg = sum(meteor_scores) / len(meteor_scores)

print("METEOR score:", meteor_score_avg)


# In[14]:


import nltk
nltk.download('wordnet')


# In[17]:


import nltk
nltk.download('omw-1.4')


# In[45]:


from nltk.translate.bleu_score import sentence_bleu

img_path = 'C:/Users/dheer/OneDrive/Desktop/practice_work/bike.jpg'

# Load the tokenizer and model
max_length = 32
tokenizer = load(open("C:/Users/dheer/OneDrive/Desktop/practice_work/tokenizer.p","rb"))
model = load_model('C:/Users/dheer/OneDrive/Desktop/practice_work/modelsss/model_9.h5')
xception_model = Xception(include_top=False, pooling="avg")

# Extract features from the input image
photo = extract_features(img_path, xception_model)
img = Image.open(img_path)

# Generate the caption
generated_sentences = generate_desc(model, tokenizer, photo, max_length)

# Convert the generated caption to a list of words
generated_sentences = generated_sentences.split()

# Use the generated caption as the reference caption
reference_sentences = [generated_sentences]

# Compute the METEOR score
meteor = meteor_score(reference_sentences, generated_sentences)

# Print the METEOR score
print("METEOR Score:", meteor)


# In[ ]:


from nltk.translate.meteor_score import meteor_score
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.xception import Xception, preprocess_input
from keras.models import load_model
import pandas as pd
import numpy as np
from pickle import load

# Load the tokenizer and model
max_length = 32
tokenizer = load(open("tokenizer.p","rb"))
model = load_model('model_9.h5')
xception_model = Xception(include_top=False, pooling="avg")

# Load the image descriptions from the CSV file
df = pd.read_csv('Book1.csv')

# Initialize lists to store reference and generated captions
references = []
generated = []

# Loop through each row in the CSV file
for index, row in df.iterrows():
    # Load the image and preprocess it
    img_path = row['image_path']
    img = load_img(img_path, target_size=(299, 299))
    img = img_to_array(img)
    img = preprocess_input(img)
    
    # Extract features from the input image
    photo = xception_model.predict(np.array([img]))
    
    # Generate the caption
    generated_sentence = generate_desc(model, tokenizer, photo, max_length)
    generated_sentence = generated_sentence.split()
    
    # Convert the ground truth caption to a list of words
    reference_sentence = row['caption'].split()
    
    # Add the reference and generated captions to the corresponding lists
    references.append([reference_sentence])
    generated.append(generated_sentence)

# Compute the METEOR score
meteor = meteor_score(references, generated)

# Print the METEOR score
print("METEOR Score:", meteor)


# In[31]:


from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from PIL import Image
from pickle import load
from keras.models import load_model
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input
from keras.preprocessing import image
import numpy as np

# Load the tokenizer and model
max_length = 32
tokenizer = load(open("C:/Users/dheer/OneDrive/Desktop/practice_work/tokenizer.p","rb"))
model = load_model('C:/Users/dheer/OneDrive/Desktop/practice_work/modelsss/model_9.h5')
xception_model = Xception(include_top=False, pooling="avg")

# Define a function to preprocess the input image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Extract features from the input image
img_path = 'C:/Users/dheer/OneDrive/Desktop/practice_work/bike.jpg'
photo = extract_features(img_path, xception_model)

# Generate the caption
generated_caption = generate_desc(model, tokenizer, photo, max_length)

# Convert the generated caption to a list of words
generated_caption = generated_caption.split()

# Use the generated caption as the reference caption
reference_captions = [generated_caption]

# Calculate the BLEU-4 score with smoothing
smoothie = SmoothingFunction().method4
bleu_score = sentence_bleu(reference_captions, generated_caption, smoothing_function=smoothie, weights=(0.25, 0.25, 0.25, 0.25))

# Print the BLEU-4 score
print("BLEU-4 Score:", bleu_score)


# In[47]:


# meteor*100


# In[35]:


pip install transformers


# In[1]:


from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("ml6team/keyphrase-extraction-distilbert-inspec")

model = AutoModelForTokenClassification.from_pretrained("ml6team/keyphrase-extraction-distilbert-inspec")


# In[23]:


import nltk
nltk.download('averaged_perceptron_tagger')


# In[25]:


nltk.download('maxent_ne_chunker')


# In[27]:


nltk.download('words')


# In[4]:


import pickle

# Load extracted features from pickle file
features_file = "C:/Users/dheer/OneDrive/Desktop/practice_work/features.p"  # Update with your pickle file path
with open(features_file, "rb") as f:
    features = pickle.load(f)

# Access and view the extracted features
for img_filename, feature_vector in features.items():
    print("Image Filename:", img_filename)
    print("Feature Vector Shape:", feature_vector.shape)
    print("Feature Vector:", feature_vector)
    print("---------")


# In[5]:


import pickle

# Load extracted features from pickle file
features_file = "C:/Users/dheer/OneDrive/Desktop/practice_work/features.p"  # Update with your pickle file path
with open(features_file, "rb") as f:
    features = pickle.load(f)

# Access and view the extracted features and their values
for img_filename, feature_vector in features.items():
    print("Image Filename:", img_filename)
    print("Features:")
    for i, feature_value in enumerate(feature_vector.flatten()):
        print(f"Feature {i+1}: {feature_value}")
    print("---------")


# In[ ]:




