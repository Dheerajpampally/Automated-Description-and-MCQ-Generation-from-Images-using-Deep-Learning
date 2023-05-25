# Automated-Description-and-MCQ-Generation-from-Images-using-Deep-Learning
Developed the combination of advanced deep learning  algorithms, natural language processing techniques, key phrase identification, validating the effectiveness and accuracy of the proposed approach in automated description and question generation from images deployed as a web application using Flask framework.

# Dataset : Flicker8k_Datasets

<a href="https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip">Flicker8K_Dataset</a> - Contains 8092 images in jpeg format.<br>
<a href="https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip">Flicker8k_Text</a> - Each image contains 5 description.<br>
or <br>
<a href="https://www.kaggle.com/datasets/ming666/flicker8k-dataset">download from kaggle</a>

# Model
  <li>Image features are extracted using Xception Model</li><br>
  <li>Used sequential CNN and LSTM model to combine to build Image description model<li><br>

# Generate fill-in-the blanks MCQ from generated Description
  <li>Used POS tagging to identify the entities and then replaced with a blank space<li><br>
  <li>fastText model implemented in genism is used to generate answer options similar to the entities<li><br>
