1. Prepare photo and text data for training a deep learning model.


2. Design and train a deep learning caption generation model.


3. Evaluate a train caption generation model and use it to caption entirely new photographs.


###### Dataset:
    Flickr8k_Dataset.zip (1 Gigabyte) An archive of all photographs.
    Flickr8k_text.zip (2.2 Megabytes) An archive of all text descriptions for photographs.


    Flickr8k_Dataset: Contains 8092 photographs in JPEG format.
    Flickr8k_text: Contains a number of files containing different sources of descriptions for the photographs.


The dataset has a pre-defined training dataset (6,000 images), development dataset (1,000 images), and test dataset (1,000 images).


One measure that can be used to evaluate the skill of the model are BLEU scores

###### Dataset Request form. https://forms.illinois.edu/sec/1713398



We will use a pre-trained model to interpret the content of the photos. In this case, it is VGG model that won the ImageNet competition in 2014.

----Keras provides this pre-trained model directly. Note, the first time you use this model, Keras will download the model weights from the Internet, which are about 500 Megabytes----


We pre-compute the “photo features” using the pre-trained model and save them to file.

Load these features later and feed them into our model as the interpretation of a given photo in the dataset.

Remove the last layer from the loaded model, as this is the model used to predict a classification for a photo. We are interested in the internal representation of the photo right before a classification is made. These are the “features” that the model has extracted from the photo.

**extract_features()** that, given a directory name, will load each photo, prepare it for VGG, and collect the predicted features from the VGG model. The image features are a 1-dimensional 4,096 element vector.



###### Prepare Text Data

The dataset contains multiple descriptions for each photograph and the text of the descriptions requires some minimal cleaning.

**load_descriptions()** - given the loaded document text, will return a dictionary of photo identifiers to descriptions.




###### Clean the description text

clean the text in the following ways in order to reduce the size of the vocabulary of words we will need to work with:

**    Convert all words to lowercase.**
**    Remove all punctuation.**
**    Remove all words that are one character or less in length (e.g. ‘a’).**
**    Remove all words with numbers in them.**
**


**clean_descriptions()** function that, given the dictionary of image identifiers to descriptions, steps through each description and cleans the text.

Ideally, we want a vocabulary that is both expressive and as small as possible. A smaller vocabulary will result in a smaller model that will train faster.


We transform the clean descriptions into a set and print its size to get an idea of the size of our dataset vocabulary.


**save_descriptions()** - Finally, we can save the dictionary of image identifiers and descriptions to a new file named descriptions.txt, with one image identifier and description per line.






###### Develop Deep Learning Model

In this section, we will define the deep learning model and fit it on the training dataset.

This section is divided into the following parts:
**
    Loading Data.
    Defining the Model.
    Fitting the Model.
**

The model we will develop will generate a caption given a photo, and the caption will be generated one word at a time. The sequence of previously generated words will be provided as input. Therefore, we will need a ‘first word’ to kick-off the generation process and a ‘last word‘ to signal the end of the caption.

We will use the strings ‘startseq‘ and ‘endseq‘ for this purpose. These tokens are added to the loaded descriptions as they are loaded. It is important to do this now before we encode the text so that the tokens are also encoded correctly.

The description text will need to be encoded to numbers before it can be presented to the model as in input or compared to the model’s predictions. ie - create a consistent mapping from words to unique integer values.

Keras provides the **Tokenizer class** that can learn this **mapping from the loaded description data**.



Each description will be split into words. The model will be provided one word and the photo and generate the next word. Then the first two words of the description will be provided to the model as input with the image to generate the next word. This is how the model will be trained.


![Screenshot 2019-07-13 at 8 12 12 PM](https://user-images.githubusercontent.com/24625231/61174776-8ed3e280-a5c2-11e9-8cb0-6f2e2a187ec2.jpg)




	
X1,     X2 (text sequence), &nbsp;   &nbsp;   &nbsp;   &nbsp;   &nbsp;   &nbsp;   &nbsp;	(word)<br/>
photo	startseq, &nbsp;   &nbsp;   &nbsp;   &nbsp;   &nbsp;   &nbsp;   &nbsp;   &nbsp; 	little<br/>
photo	startseq, little,  &nbsp;   &nbsp;   &nbsp;   &nbsp;   &nbsp; 				girl<br/>
photo	startseq, little, girl, &nbsp;   &nbsp;   &nbsp;   &nbsp;   &nbsp; 			running<br/>
photo	startseq, little, girl, running, &nbsp;   &nbsp;   &nbsp;   &nbsp;   &nbsp; 		in<br/>
photo	startseq, little, girl, running, in, &nbsp;   &nbsp;   &nbsp;   &nbsp;   &nbsp; 	field<br/>
photo	startseq, little, girl, running, in, field, &nbsp;   &nbsp;   &nbsp;   &nbsp;   &nbsp; endseq<br/>


Later, when the model is used to generate descriptions, the generated words will be concatenated and recursively provided as input to generate a caption for an image.


The function named **create_sequences()**, given the tokenizer, a maximum sequence length, and the dictionary of all descriptions and photos, will transform the data into input-output pairs of data for training the model. There are two input arrays to the model: one for photo features and one for the encoded text. There is one output for the model which is the encoded next word in the text sequence.

The **input text is encoded as integers, which will be fed to a word embedding layer**. The photo features will be fed directly to another part of the model. The model will output a prediction, which will be a probability distribution over all words in the vocabulary.

The output data will therefore be a one-hot encoded version of each word, representing an idealized probability distribution with 0 values at all word positions except the actual word position, which has a value of 1.



We will describe the model in three parts:
    **Photo Feature Extractor** -  This is a 16-layer VGG model pre-trained on the ImageNet dataset. We have pre-processed the photos with the VGG model (without the output layer) and will use the extracted features predicted by this model as input.
    **Sequence Processor** -  This is a word embedding layer for handling the text input, followed by a Long Short-Term Memory (LSTM) recurrent neural network layer.
    **Decoder (for lack of a better name)** -  Both the feature extractor and sequence processor output a fixed-length vector. These are merged together and processed by a Dense layer to make a final prediction.
    **Embedding layer** :  [https://stats.stackexchange.com/questions/270546/how-does-keras-embedding-layer-work](https://stats.stackexchange.com/questions/270546/how-does-keras-embedding-layer-work) 
             **LSTM** :    [https://stackoverflow.com/questions/53966446/lstm-architecture-in-keras-implementation](https://stackoverflow.com/questions/53966446/lstm-architecture-in-keras-implementation)

   **Embedding(vocab_size, 256, mask_zero=True)**
   **LSTM(256)**

   Embedding table is 
   
![Screenshot 2019-07-13 at 8 12 05 PM](https://user-images.githubusercontent.com/24625231/61174770-6f3cba00-a5c2-11e9-8615-3d2ec0c3ef11.jpg)


Embedding : 
	Every word is associated with an index by order of appearance in our training dataset.
	The vocab_size indicate the length of this table.(No. of words whose vector is to be created)
	Length of each vector is 256 in the above case.

LSTM :
	Length of output vector from each LSTM cell is 256.
	Total LSTM cells are equal (33/34) to the length of the sentence with maximum words.

The LSTM layer is used as an encoder layer meaning that it is used as an input layer to take the initial part of the caption(sequence).

Decoder:
	The Decoder layer is used as the output layer which is used to generate the predicted next word given the initial part of the sequence and the CNN combined.

![Screenshot 2019-07-13 at 11 18 18 PM](https://user-images.githubusercontent.com/24625231/61174914-8a102e00-a5c4-11e9-90a1-3dbc37fdade2.jpg)




![Screenshot 2019-07-13 at 8 12 19 PM](https://user-images.githubusercontent.com/24625231/61174778-92676980-a5c2-11e9-8f31-750aacf73f89.jpg)

