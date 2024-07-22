from database import Database
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet
import mysql.connector

# # Connect to your MySQL database
# mydb = mysql.connector.connect(
#     host="localhost",
#     user="root",
#     password=" ",
#     database="chatbot_database"
# )

# # Fetch data from the intents table
# mycursor = mydb.cursor()
# mycursor.execute("SELECT tag, patterns, responses FROM intents")
# intents_data = mycursor.fetchall()




# Initialize Database instance
db = Database(host="localhost", user="root", password="", database="chatbot_database")
# Fetch data from the database
intents_data = db.fetch_intents_data()
# Process the fetched data and train the model




# Process the fetched data
all_words = []
tags = []
xy = []
for row in intents_data:
    tag = row[0]
    tags.append(tag)
    patterns = row[1].split(';')
    for pattern in patterns:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

# Stem and lower each word
ignore_words = ['?', '.', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '+', '-', '_', '=', '"', '~', '`']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# Create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyperparameters
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

# Custom dataset class
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the trained model
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'Training complete. Model saved to {FILE}')






















































# """This file have functions which change the pre-defined model into the chatbot-model"""

# import numpy as np
# import random
# import json
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader

# from nltk_utils import bag_of_words, tokenize, stem
# from model import NeuralNet

# with open('intents.json', 'r') as f:
#     intents = json.load(f)

# all_words = []
# tags = []
# xy = []
# # loop through each sentence in our intents patterns
# for intent in intents['intents']:
#     tag = intent['tag']
#     # add to tag list
#     tags.append(tag)
#     for pattern in intent['patterns']:
#         # tokenize each word in the sentence
#         w = tokenize(pattern)
#         # add to our words list
#         all_words.extend(w)
#         # add to xy pair
#         xy.append((w, tag))

# # stem and lower each word
# ignore_words = ['?', '.', '!','@','#','$','%','^','&','*','(',')','+','-','_','=','"','~','`']
# all_words = [stem(w) for w in all_words if w not in ignore_words]
# # remove duplicates and sort
# all_words = sorted(set(all_words))
# tags = sorted(set(tags))

# print(len(xy), "patterns")
# print(len(tags), "tags:", tags)
# print(len(all_words), "unique stemmed words:", all_words)

# # create training data
# X_train = []
# y_train = []
# for (pattern_sentence, tag) in xy:
#     # X: bag of words for each pattern_sentence
#     bag = bag_of_words(pattern_sentence, all_words)
#     X_train.append(bag)
#     # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
#     label = tags.index(tag)
#     y_train.append(label)

# X_train = np.array(X_train)
# y_train = np.array(y_train)

# # Hyper-parameters 
# num_epochs = 1000
# batch_size = 8
# learning_rate = 0.001
# input_size = len(X_train[0])
# hidden_size = 8
# output_size = len(tags)
# print(input_size, output_size)

# class ChatDataset(Dataset):

#     def __init__(self):
#         self.n_samples = len(X_train)
#         self.x_data = X_train
#         self.y_data = y_train

#     # support indexing such that dataset[i] can be used to get i-th sample
#     def __getitem__(self, index):
#         return self.x_data[index], self.y_data[index]

#     # we can call len(dataset) to return the size
#     def __len__(self):
#         return self.n_samples

# dataset = ChatDataset()
# train_loader = DataLoader(dataset=dataset,
#                           batch_size=batch_size,
#                           shuffle=True,
#                           num_workers=0)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = NeuralNet(input_size, hidden_size, output_size).to(device)

# # Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# # Train the model
# for epoch in range(num_epochs):
#     for (words, labels) in train_loader:
#         words = words.to(device)
#         labels = labels.to(dtype=torch.long).to(device)
        
#         # Forward pass
#         outputs = model(words)
#         # if y would be one-hot, we must apply
#         # labels = torch.max(labels, 1)[1]
#         loss = criterion(outputs, labels)
        
#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#     if (epoch+1) % 100 == 0:
#         print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# print(f'final loss: {loss.item():.4f}')

# data = {
# "model_state": model.state_dict(),
# "input_size": input_size,
# "hidden_size": hidden_size,
# "output_size": output_size,
# "all_words": all_words,
# "tags": tags
# }

# FILE = "data.pth"
# torch.save(data, FILE)

# print(f'training complete. file saved to {FILE}')
