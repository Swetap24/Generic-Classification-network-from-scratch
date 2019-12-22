# Generic-Classification-network-from-scratch
ResNet Model for classification task
Classification is nothing but predicting which class an item belongs to. Some classifiers could be binary(Spam/no Spam, Defect/No defect) while others are multi-class. Classification is a very common use case of deep learning where classification algorithms are used to solve problems like email spam filtering, document categorization, speech recognition, image recognition, biomdeical usage and handwriting recognition.

Classification network could be developed in many ways using different neural network or ML algorithms but the one that I propose is based on ResNet Network. This is a very generic code and could be implemented for any classification task if you have your own dataset and the number of classes to be detected is known. I hope this helps many student/developers develop different classification models for various applications. This does not use any pre-defined model or transfer learning hence, could be used for any classification task from scratch.




This is my first public repository!
Do give feedback if any :)





# Dataloader-
Replace the name of train and validation and test dataset with your own dataset.

# Data preparation-
Within a train folder make folder for each class and put particular class data in that respective folder.
Example
Train folder-->>/spam /nospam-->>> /spam folder contain all the data for spam dataset -->>>/nospam folder contain all the data for nospam dataset
Repeat the same for train, validation and test folder.
# Note: Do not repeat same dataset in train, validate and test folder. If dataset is less, please try using k-fold cross validation to obtain validation set.
I'll try to make a separate post for the same :)

In ResNet block, number of classes could be changed to number of classes you have in your classification task(need not be 2 in multi- class classification)

Just specify path to save model and you are good to go :)

Happy Coding!
