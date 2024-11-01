import sys
import numpy as np
import torch

from torch.nn import Module, Linear, Embedding, NLLLoss, CrossEntropyLoss
from torch.nn.functional import relu, log_softmax
from torch.utils.data import Dataset, DataLoader 

"""
  3. Designing the network

  specify and train the neural network model
  writes a file containing the model architecture and trained weights

  -> 4. Training loop and running training

  ##### MODIFY REQUIRED

  >> python train_model.py data/input_train.npy data/target_train.npy data/model.pt
  In my experiments, after 5 epochs I reached a training loss of < 0.31 and a training accuracy of about 0.90.

  My model
    Training loss epoch: 0.3084476339585946,   Accuracy: 0.9022273421287537
"""

from extract_training_data import FeatureExtractor

class DependencyDataset(Dataset):

  def __init__(self, input_filename, output_filename):
    self.inputs = np.load(input_filename)
    self.outputs = np.load(output_filename)

  def __len__(self): 
    return self.inputs.shape[0]

  def __getitem__(self, k): 
    return (self.inputs[k], self.outputs[k])


class DependencyModel(Module): 

  def __init__(self, word_types, outputs):
    super(DependencyModel, self).__init__()
    ##### TODO: complete for part 3
    """
      [Structure]
        input (batch, 6)
        concatenated embeddings (batch, (6*128))
        hidden layer (batch, 128 units) relu activated
        output layer (batch, 91 units) softmax activated
    """

    """
      Embedding layer
        input: stack & buffer tokens
        params: word_types, embedding_dim = 128
    """
    self.embedding_dim = 128
    self.embedding = Embedding(num_embeddings=word_types, embedding_dim=self.embedding_dim)

    """
      Linear layer
    """
    input_num = 6
    hidden_num = 128
    output_num = 91

    self.fc1 = Linear(input_num*self.embedding_dim, hidden_num)
    self.fc2 = Linear(hidden_num, outputs)

  def forward(self, inputs):

    ##### TODO: complete for part 3
    inputs = inputs.type(torch.LongTensor)
    embeddings = self.embedding(inputs) # input -> embedding
    embeddings_flatten = embeddings.view(embeddings.size(0), -1) # flatten
    hidden = relu(self.fc1(embeddings_flatten)) # embedding -> hidden
    output = log_softmax(self.fc2(hidden)) # hidden -> output [shape:(batch, 91)]

    # return torch.zeros(inputs.shape(0), 91)  # replace this line
    return output


def train(model, loader): 

  loss_function = CrossEntropyLoss(reduction='mean')

  LEARNING_RATE = 0.01 
  optimizer = torch.optim.Adagrad(params=model.parameters(), lr=LEARNING_RATE)

  tr_loss = 0 
  tr_steps = 0

  # put model in training mode
  model.train()
 

  correct = 0 
  total =  0 
  for idx, batch in enumerate(loader):
 
    inputs, targets = batch
 
    predictions = model(torch.LongTensor(inputs))

    loss = loss_function(predictions, targets)
    tr_loss += loss.item()

    #print("Batch loss: ", loss.item()) # Helpful for debugging, maybe 

    tr_steps += 1
    
    if idx % 1000==0:
      curr_avg_loss = tr_loss / tr_steps
      print(f"Current average loss: {curr_avg_loss}")

    # To compute training accuracy for this epoch 
    correct += sum(torch.argmax(predictions, dim=1) == torch.argmax(targets, dim=1))
    total += len(inputs)
      
    # Run the backward pass to update parameters 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


  epoch_loss = tr_loss / tr_steps
  acc = correct / total
  print(f"Training loss epoch: {epoch_loss},   Accuracy: {acc}")


if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r')
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1)

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)


    model = DependencyModel(len(extractor.word_vocab), len(extractor.output_labels))

    dataset = DependencyDataset(sys.argv[1], sys.argv[2])
    loader = DataLoader(dataset, batch_size = 16, shuffle = True)

    print("Done loading data")

    # Now train the model
    for i in range(5): 
      train(model, loader)


    torch.save(model.state_dict(), sys.argv[3]) 
