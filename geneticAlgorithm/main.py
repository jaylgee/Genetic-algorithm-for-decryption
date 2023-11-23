# TO DO:
# Import the encryption message
# Create a function to create the guess of the message using a guess cipher
# use that 'guess' string of the message as 'cipher_guess_string' at line 84 to get its log likelihood
# then, need to do the mutate (pair switch) and calc the log likelihood
# Then need to continue this as a recursion until a maximum found or a max iters reached
# Also need to seed an initial population of guesses (ie more than one)
# Take the best log likelihood each time.


import random
import copy
import re
import math
import numpy as np


def main():

    # get the Moby Dick text for the language model
    # corpus = requests.get('https://lazyprogrammer.me/course_files/moby_dick.txt')
    # create a file object from the text in binary and decode it with utf-8 encoding
    with open("moby_dick.text", "rb") as f:
        corpus_clean = f.read().decode('utf-8')

    # import the sherlock holmes test message
    test_message = "I then lounged down the street and found, as I expected, " \
                   "that there was a mews in a lane which runs down by one wall " \
                   "of the garden. I lent the ostlers a hand in rubbing down their " \
                   "horses, and received in exchange twopence, a glass of half-and-half, " \
                   "two fills of shag tobacco, and as much information as I could desire " \
                   "about Miss Adler, to say nothing of half a dozen other people in " \
                   "the neighbourhood in whom I was not in the least interested, but whose " \
                   "biographies I was compelled to listen to."

    # remove non alpha characters
    corpus_clean = "".join([char for char in corpus_clean if char.isalpha() or char.isspace()])
    # replace ae joins with a and e
    corpus_clean = re.sub(chr(230), chr(97), corpus_clean)
    corpus_clean = re.sub(chr(233), chr(101), corpus_clean)
    corpus_clean = re.sub(chr(232), chr(101), corpus_clean)
    corpus_clean = re.sub(chr(339), chr(101), corpus_clean)
    corpus_clean = re.sub(chr(226), chr(97), corpus_clean)
    #print(corpus_clean)

    # tokenise the texts into a word list?
    tokens = corpus_clean.lower().split()  # training text
    test_message = re.sub(r'[^\w\s]', '', test_message)
    test_tokens = test_message.lower().split()  # test text

    #print(test_tokens)

    # generate a random substitution cipher as a dict mapping clear to encrypted values
    true_cipher = generate_true_cipher()  # this is the key to the message that the language model tries to find
    encrypted_solution = encode_text(test_tokens, true_cipher)  # this is the message that we're trying to decode
    print(f"encrypted message is {encrypted_solution}")
    # print(true_cipher)
    # print(encrypted_solution)

    # create a character level language model
    # numpy vector of unigrams (first characters) and bigrams (two chars)
    prob_firsts = np.zeros(26)
    prob_two_chars = np.ones((26, 26))
    # count all the first letters of the tokens

    # update prob_firsts. Add one to the count corresponding to that letter
    def update_prob_firsts(ch):
        letter = ord(ch.lower()) - 97
        #print(f"ch is {ch}, letter is {letter}")
        #if letter > 25 or letter < 0:
        #    print(f"found error in {ch}")
        prob_firsts[letter] += 1

    # update array of bigrams, prob_two_chars, adding one each time bigram occurs.
    def update_prob_two_chars(ch1, ch2):
        i = ord(ch1.lower()) - 97
        j = ord(ch2.lower()) - 97
        prob_two_chars[i, j] += 1

    # loop through tokens and update the arrays with the counts
    for word in tokens:
        #print(word)
        #print(letter_to_nums(word))
        # add count for first letter in pi array (prob_firsts)
        # first letter
        char0 = word[0:1]
        #print(f"char0 is {char0}")
        update_prob_firsts(char0)
        # other letters
        for char1 in word[1:]:
            # add count for bigrams in transition matrix (prob_two_chars)
            update_prob_two_chars(char0, char1)
            char0 = char1

    # normalise the counts
    prob_firsts = prob_firsts/prob_firsts.sum()
    prob_two_chars = prob_two_chars/prob_two_chars.sum(axis=1, keepdims=True)
    #print(np.sum(prob_two_chars, axis=1).tolist()) //returns a list of sums of rows

    # run the genetic algorithm
    # set an initial population size
    initial_population_size = 40
    # set maximum number of iterations
    max_iterations = 500

    # create dna_pool of initial population size
    dna_pool = []
    top_performers_likelihood_list = [] # should this be a dict {dna: log_likelihood, dna1: log_likelihood...}
    top_performers = {}
    closest_fit_dna = {}
    max_likelihood = float('-inf')
    # generate an initiating cipher for the dna pool
    for i in range(initial_population_size):
        # call generate cipher function
        dna = generate_true_cipher()  # dictionary that will act as the first parents
        dna_pool.append(dna)
    #print(dna_pool)


    for i in range(max_iterations):
        # create the children for the dna_pool
        #print(f"before children, dna pool is {dna_pool}")
        dna_pool = children_generator(dna_pool, 3)  # list of dictionaries
        # calc the log likelihood of all the dna sequences and take top performers
        for dna in dna_pool:
            # decode the encrypted message using the dna cipher
            decoded_message = encode_text(encrypted_solution, dna)
            dna_likelihood = log_likelihood(decoded_message, prob_firsts, prob_two_chars)
            # print(i, dna_likelihood)
            #if i % 100 == 0:
            #    print(i, dna_likelihood)
            # find the worst performer among the best
            try:
                worst_of_the_best = min(top_performers_likelihood_list)  # might need to handle an empty list here
            except ValueError:
                worst_of_the_best = float('-inf')
            # add to best performers if new max_likelihood
            if dna_likelihood > max_likelihood:
                # replace the max_likelihood
                max_likelihood = dna_likelihood
                # make this dna the 'closest fit' dna
                closest_fit_dna = dna
                # add the dna to the top performers list of dicts
                top_performers[dna_likelihood] = dna  # add the new best guess to top performers dict
                top_performers_likelihood_list.append(dna_likelihood)  # this will allow me easily find the min value
                #print(f"at a new max, top likelihood list is {top_performers_likelihood_list}")
            # add to top performers if dna_likelihood < max but still > min(top performers)
            if max_likelihood > dna_likelihood:
                if dna_likelihood > worst_of_the_best:
                    if worst_of_the_best != float('-inf'):
                        # add to the top performers list of dicts if not already in there
                        if dna_likelihood in top_performers:
                            continue
                        else:
                            top_performers[dna_likelihood] = dna
                            top_performers_likelihood_list.append(dna_likelihood)
                            #print(f"in elif, top likelihood list is {top_performers_likelihood_list}")
            # adjust number of top performers and the likelihood list if required
            if len(top_performers_likelihood_list) > 20:
                # top_performers.remove(min(top_performers)) # is this doing what I need?

                # remove dna with lowest log likelihood from top performers dict
                #print(f"worst of the best is {worst_of_the_best}")
                # remove lowest scorer from top performers list
                #print(f"length top_performers before = {len(top_performers)}")
                #print(f"length top likelihood list before is {len(top_performers_likelihood_list)}")
                del top_performers[worst_of_the_best]


                # update the top performers likelihood list

                #print(f"removing {worst_of_the_best}")
                top_performers_likelihood_list.remove(worst_of_the_best)
                #print(f"length likelihood after is  {len(top_performers_likelihood_list)}")
                #print(f"length top_performers after = {len(top_performers)}")
        # initialise for next generation
        # replace the dna pool with the top 5 performers and continue the iterations
        # dna_pool is a list of dicts. top performers is a dict of dicts

        dna_pool = []  # empty the original dict
        top_performers_likelihood_list = []  # empty the list
        for k, v in top_performers.items():
            # add to the dna pool
            dna_pool.append(v)
            # add to the top performers likelihood list
            top_performers_likelihood_list.append(k)
        #print(f"top performers are: {top_performers}")

    # return the best solution
    print(f"max likelihood is {max_likelihood}")
    print(f"message with closest fit dna is:\n{encode_text(encrypted_solution, closest_fit_dna)}")


# function to evolve the dna pool
def children_generator(spawn_pool, no_of_children_to_generate):
    dna_children = []
    # create the dna mutations (ie the children) by switching two of the codes
    for dna in spawn_pool:  # dna_pool is a list of dictionaries of numbers
        # create n children per parent dna:
        for i in range(no_of_children_to_generate):
            # switch two values around
            num_one = np.random.randint(26)  # creates a random int, which will be one of the values to swap
            num_two = np.random.randint(26)  # creates a random int, which will be one of the values to swap
            # handle num one = num two, which is unwanted as won't swap
            while num_two == num_one:
                num_two = np.random.randint(26)
            #print(num_one, num_two)
            # switch the values for the keys corresponding to the random ints generated above
            # create a copy of the dictionary to mutate it
            child_dna = copy.deepcopy(dna)
            child_dna[num_one] = dna[num_two]
            child_dna[num_two] = dna[num_one]
            #print(dna)
            #print(child_dna)
            dna_children.append(child_dna)
    spawn_pool += dna_children
    #print(f"dna pool size with children after mutation is {len(spawn_pool)}")
    return spawn_pool


# function to encode text
def encode_text(list_of_strings, cipher_key):
    #print(list_of_strings)
    # convert list_of_strings to nums
    num_list = []
    encrypted_word_list = []

    for item in list_of_strings:
        num_word = []
        #print(item)
        word_as_num = letter_to_nums(item)
        #print(word_as_num)
        #num_list.append(word_as_num)
        for num in word_as_num:
            encrypted_num = cipher_key[num]
            num_word.append(encrypted_num)
        num_list.append(num_word)

    # convert back to a list of strings
    for item in num_list:
        new_string = ""
        #print(item)
        for i in range(len(item)):
            #print(item[i])
            new_string += chr(item[i] + 97)  # this need to return a letter
        #print(new_string)
        encrypted_word_list.append(new_string)

    return encrypted_word_list


# function to take in a list of strings and generate its log likelihood
# f(message) = log likelihood
def log_likelihood(list_of_strings, pi, transition):
    # for each string in the list, calc pi and ptransition
    # p(list of strings) = log p(w1) + log p(w2) etc
    # log p(w1char1) + log p(w1char2 | w1 char 1) etc.
    log_likelihood_counter_word = 0
    log_likelihood_counter_sentence = 1

    # TO DO: check the logic here
    for my_string in list_of_strings:
        # get my_string as a list of ascii numbers
        num_list = letter_to_nums(my_string)
        # get log likelihood of first char
        ch0 = num_list[0]
        # add pi log likelihood to counter
        log_likelihood_counter_word += math.log(pi[ch0])
        # deal with remaining chars
        for ch1 in num_list[1:]:
            #print(ch1)
            log_likelihood_counter_word += math.log(transition[ch0, ch1])
            ch0 = ch1
        # add word likelihood to the sentence counter
        log_likelihood_counter_sentence += log_likelihood_counter_word
        # reset word likelihood count to zero
        log_likelihood_counter_word = 0

    return log_likelihood_counter_sentence


# convert tokens to numbers
def letter_to_nums(my_token):
    #print(my_token)
    my_token = my_token.lower()
    converted_list = []
    for i in range(len(my_token)):
        letter = my_token[i: i+1]
        result = ord(letter) - 97
        if result > 25 or result < 0:
            print(f"found error in {my_token}")
        converted_list.append(result)
    return converted_list


def generate_true_cipher():
    # create random list of numbers between 0 and 25
    my_list = []
    for i in range(26):
        my_list.append(i)
    # print(my_list)
    new_list = copy.deepcopy(my_list)
    # shuffle the list
    random.shuffle(new_list)
    # function to generate cipher as letters from the numbers

    def cipher_func(a, b):
        return a, b
    # map the numeric values into a dictionary
    x = dict(map(cipher_func, my_list, new_list))

    return x  # returns a dict


# convert numeric cypher into chars
def cipher_text_func(a, b):
    return chr(a+97).lower(), chr(b+97).lower()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
