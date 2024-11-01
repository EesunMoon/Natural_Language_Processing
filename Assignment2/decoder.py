import sys
import copy

import numpy as np
import torch

from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from extract_training_data import FeatureExtractor, State
from train_model import DependencyModel

"""
    5. Greedy Parsing Algorithm - Building and Evaluating the Parser

    uses the trained model file to parse some input
    For simplicity, the input is a CoNLL-X formatted file,
    but the dependency structure in the file is ignored

    Prints the parser output for each sentence in CoNLL-X format

    ##### MODIFY REQUIRED

    >> python decoder.py data/model.pt data/dev.conll
    >> python evaluate.py data/model.pt data/dev.conll

        [Mine]
        5039 sentence.

        Micro Avg. Labeled Attachment Score: 0.7272124242348864
        Micro Avg. Unlabeled Attachment Score: 0.7742676384501215

        Macro Avg. Labeled Attachment Score: 0.738056007037474
        Macro Avg. Unlabeled Attachment Score: 0.7849849211163782

"""

class Parser(object):

    def __init__(self, extractor, modelfile):
        self.extractor = extractor

        # Create a new model and load the parameters
        self.model = DependencyModel(len(extractor.word_vocab), len(extractor.output_labels))
        self.model.load_state_dict(torch.load(modelfile))
        sys.stderr.write("Done loading model")

        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):

        state = State(range(1,len(words))) 
        state.stack.append(0) # stack: starts with root(index = 0)

        ##### TODO: Write the body of this loop for part 5
        while state.buffer:
            features = self.extractor.get_input_representation(words, pos, state)
            transition_pred = self.model(torch.tensor([features], dtype=torch.long))
            sorted_actions = np.argsort(transition_pred[0].detach().numpy())[::-1] ## highest first
            
            for idx in sorted_actions:
                transition, label = self.output_labels[idx]

                if transition == 'shift':
                    if len(state.buffer) > 1 or len(state.stack) ==0:
                        state.shift()
                        break
                else:
                    if len(state.stack) !=0:
                        ## ROOT pass: skip left_arc if stack's top is root (index = 0)
                        if transition == 'left_arc' and state.stack[-1] !=0:
                            state.left_arc(label)
                            break
                        elif transition == 'right_arc':
                            state.right_arc(label)
                            break
    

        result = DependencyStructure()
        for p,c,r in state.deps:
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))

        return result


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
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file:
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
