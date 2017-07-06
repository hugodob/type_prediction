# type_prediction
Predicting the type of a field depending on its display name

The objective of this project is: given a data-set with display names associated with fields and their types, try to predict the type of a field depending on its display name.<br />
There are 4 files which result from the 4 main stages I had during this project. <br />

> ### DataSet.py<br />
This file defines a new object called DataSet which enables us to extract data from a csv, clean this data according to what we want to do with it and plot some indicators which will help us driving our timeline. <br />
I used it in order to clean the data set and ploted the 10 most common types. This made me realise that the types "Folder" and "Tags" were overused and we finally decided not to take them into account for the predictions. Better still, ploting the data also showed that a lot of custom types were used but each one only a few times. These types being registered directly by the customer, we decided not to take them into account as we couldn't predict what doesn't already exist. <br />
This is why, we finally we decided to deal with these types: ['STRING', 'TEXT', 'PERSON', 'DATE', 'INTEGER', 'BOOLEAN', 'DECIMAL', 'DATETIME', 'URL'], giving us a 24,000 rows data set after all clearings. <br />
![](graph_first_data_exploration/10_most_common_type.png)
Here are the 10 most common types having cleared "Tags" and "Folder": <br/><br/>


> ### vectorization_Doc2Vec.py <br />
After formatting the data as mentioned above, we end up with a matrix containing all our display names in its first column and all the types associated in the second one. The problem here, is that we want to classify the display names and would therefore like to use a multilayer perceptron, however the input data should be float vectors and not string ones. Consequently we need to convert our display names to vectors and this is the tricky part. <br/>
A first implementation would be to use the "bag of words" method which would convert each display name to a vector by adding the vectors associated to each word. However, this method wouldn't take into consideration the similarity between some names and for example the vectors associated with "closing date" and "sale's date" could be very different one to the other, while here we would like them to be close as both are going to end up with a "Date" type. <br/>
Therefore, we decided to use the Doc2Vec model from gensim, which takes all our inputs, builds a dictionnary based on the documents (here the display names) we gave it and associates better suited vectors to each display name. 


