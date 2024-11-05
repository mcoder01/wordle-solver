import os
import argparse
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

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

def parallel(size, target):
    with ThreadPoolExecutor() as executor:
        num_threads = executor._max_workers
        N = size//num_threads
        carry = size%num_threads

        lock = Lock()
        futures = []
        for i in range(num_threads):
            nloc = N+1 if i < carry else N
            start = i*N + min(i, carry)
            futures.append(executor.submit(target, start, nloc, lock))

        for future in futures:
            future.result()

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

def most_probable(scores):
    max = np.argmax(list(scores.values()))
    return list(scores.keys())[max]

def predict(scores):
    letters = dict()
    for word in scores.keys():
        unique_letters = []
        for letter in word:
            if letter not in unique_letters:
                unique_letters.append(letter)
                letters.setdefault(letter, 0)
                letters[letter] += 1

    letters = dict(sorted(letters.items(), key=lambda item: item[1], reverse=True))
    for letter in letters:
        new = select(scores, lambda word, _: letter in word)
        if len(new) == 0:
            break
        if len(new) == 1:
            return list(new.keys())[0]
        scores = new
    return most_probable(scores)

def guess_word(scores):
    prediction = most_probable(scores)
    for i in range(6):
        scheme = input(f"My prediction is {prediction.upper()}\nWhat is the resulting scheme? ")
        scores = prune_words(scores, prediction, scheme)
        if len(scores) == 1 or i == 5:
            print("My final answer is: " + list(scores.keys())[0])
            break
        prediction = predict(scores)
        
def test(scores, percentage=0.2):
    #testset = select(scores, lambda x, y: np.random.rand() < percentage)
    testset = scores
    guessed = 0
    attempts_mean = 0
    def guess(start, length, lock):
        nonlocal guessed, attempts_mean
        local_guessed = 0
        local_attempts = 0
        for i in range(start, start+length):
            secret = list(testset.keys())[i]
            filtered = scores.copy()
            prediction = most_probable(filtered)
            attempt = 1
            while attempt < 6:
                scheme = get_scheme(secret, prediction)
                filtered = prune_words(filtered, prediction, scheme)
                if len(filtered) == 1:
                    break
                prediction = predict(filtered)
                attempt += 1

            if len(filtered) == 1:
                local_guessed += 1
                local_attempts += attempt

        with lock:
            guessed += local_guessed
            attempts_mean += local_attempts
    
    parallel(len(testset), guess)
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