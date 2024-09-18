import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

def occurrences(word):
    occ = dict()
    for c in word:
        occ.setdefault(c, 0)
        occ[c] += 1
    return occ

def get_scheme(secret, word):
    scheme = ['0']*len(secret)
    occ = occurrences(secret)
    for i, (c1, c2) in enumerate(zip(secret, word)):
        if c1 == c2:
            scheme[i] = '2'
            occ[c1] -= 1
    
    for i, c in enumerate(word):
        if scheme[i] == '0' and occ.get(c, 0) > 0:
            scheme[i] = '1'
            occ[c] -= 1
    return ''.join(scheme)

def calculate_score(secret, word):
    scheme = get_scheme(secret, word)
    score = len(occurrences(word))/26.0
    for mask in scheme:
        if mask == '1' or mask == '2':
            score += 1/5.0
    return score

def compute_scores(words):
    scores = dict()
    for i in range(len(words)):
        scores.setdefault(words[i], 0)
        for j in range(i+1, len(words)):
            scores.setdefault(words[j], 0)
            score = calculate_score(words[i], words[j])
            scores[words[i]] += score
            scores[words[j]] += score
    return scores

def load_scores(words, scores_filename):
    scores = dict()
    with open(scores_filename, "r") as f:
        for word, value in zip(words, f.read().split(", ")):
            scores[word] = float(value)
    return scores
    
def save_scores(scores_filename, scores):
    with open(scores_filename, "w") as f:
        f.write(', '.join([str(value) for value in scores.values()]))

def select(scores, function):
    selected = dict()
    for word, score in scores.items():
        if function(word, score):
            selected[word] = score
    return selected

def plot(words, showAll=False, threshold=10000, window=1500):
    _, axis = plt.subplots(1, 1)
    x, y = select(words, lambda word: words[word] > threshold if not showAll else True)
    axis.bar(x, y)
    if not showAll:
        axis.set(ylim=[threshold, threshold+window])
    plt.show()

def prune_words(scores, prediction, scheme):
    return select(scores, lambda word, _: get_scheme(word, prediction) == scheme)

def prepare(words_file, scores_file):
    with open(words_file, "r") as f:
        words = f.read().splitlines()

    if os.access(scores_file, os.O_RDONLY):
        scores = load_scores(words, scores_file)
    else:
        scores = compute_scores(words)
        save_scores(scores_file, scores)
    return scores

def predict(scores):
    max = np.argmax(list(scores.values()))
    return list(scores.keys())[max]

def guess_word(scores):
    for i in range(6):
        if len(scores) == 1 or i == 5:
            print("My final answer is: " + list(scores.keys())[0])
            break
        
        prediction = predict(scores)
        scheme = input(f"My prediction is {prediction.upper()}\nWhat is the resulting scheme? ")
        scores = prune_words(scores, prediction, scheme)
        
def test(scores, percentage=0.2):
    testset, _ = select(scores, lambda x, y: np.random.rand() < percentage)
    guessed = 0
    attempts_mean = 0

    for secret in testset:
        attempt = 0
        filtered = scores.copy()
        while attempt < 6:
            attempt += 1
            prediction = predict(filtered)
            scheme = get_scheme(secret, prediction)
            filtered = prune_words(filtered, prediction, scheme)
            if len(filtered) == 1:
                break

        if len(filtered) == 1:
            guessed += 1
            attempts_mean += attempt

    print(f"Guess percentage: {guessed/len(testset)*100:.2f}% - Attempts mean: {attempts_mean/len(testset):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test", action="store_true", help="Test the solver on a random 20%% of the whole dictionary")
    args = parser.parse_args()

    scores = prepare("words", "scores")
    if args.test:
        test(scores)
    else:
        guess_word(scores)