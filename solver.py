import os
import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Lock, Value

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
    occ = occurrences(get_scheme(secret, word))
    return sum([(int(mask)+1)*count for mask, count in occ.items()])

def parallel(size, target, reduce=None):
    workers = os.cpu_count()
    with ProcessPoolExecutor() as executor:
        index = Value('i')
        lock = Lock()

        global run
        def run(): 
            local_result = None
            while True:
                with lock:
                    if index.value == size:
                        break
                    i = index.value
                    index.value += 1
                local_result = target(i, *local_result) if local_result else target(i)
            return local_result

        futures = []
        for _ in range(workers):
            futures.append(executor.submit(run))

        if reduce:
            while len(futures) > 1:
                results = len(futures)
                new_futures = []
                for i in range(0, results-1, 2):
                    new_futures.append(executor.submit(reduce, futures[i].result(), futures[i+1].result()))
                if results%2 == 1:
                    new_futures.append(futures[-1])
                futures = new_futures
            return futures[0].result()
        
        for future in futures:
            future.result()

def prune_words(words, prediction, scheme):
    return [word for word in words if get_scheme(word, prediction) == scheme]

def evaluate_word(words, i, scores):
    scores.setdefault(words[i], 0)
    for word in words[i+1:]:
        scores.setdefault(word, 0)
        score = calculate_score(words[i], word)
        scores[words[i]] += score
        scores[word] += score
    return scores

def compute_scores(words):
    global compute, combine
    def compute(i, scores=None):
        scores = dict() if scores is None else scores
        return (evaluate_word(words, i, scores),)

    def combine(scores1, scores2):
        scores1 = scores1[0]
        scores2 = scores2[0]
        scores = dict()
        for word in words:
            if word in scores1.keys() or word in scores2.keys():
                scores[word] = scores1.get(word, 0)+scores2.get(word, 0)
        return (scores,)

    return parallel(len(words), compute, combine)[0]

def prepare(words_file):
    with open(words_file, "r") as f:
        words = f.read().splitlines()
    return compute_scores(words)

def most_probable(scores):
    return max(scores.items(), key=lambda item: item[1])[0]

def predict(words):
    scores = compute_scores(words)
    return max(scores.items(), key=lambda item: item[1])[0]

def guess_word(scores):
    prediction = most_probable(scores)
    for i in range(6):
        scheme = input(f"My prediction is {prediction.upper()}\nWhat is the resulting scheme? ")
        scores = prune_words(scores, prediction, scheme)
        if len(scores) == 1 or i == 5:
            print("My final answer is: " + list(scores.keys())[0])
            break
        prediction = predict(scores)
        
def test(words):
    global guess, reduce
    def guess(i, local_guessed=None, local_attempts=None):
        secret = words[i]
        filtered = words.copy()
        prediction = "soare"
        attempt = 1
        while attempt < 6:
            scheme = get_scheme(secret, prediction)
            filtered = prune_words(filtered, prediction, scheme)
            if len(filtered) == 1:
                break
            prediction = predict(filtered)
            attempt += 1

        if len(filtered) == 1:
            local_guessed = local_guessed+1 if local_guessed else 1
            local_attempts = local_attempts+attempt if local_attempts else attempt
        return (local_guessed, local_attempts)

    def reduce(res1, res2):
        return res1[0]+res2[0], res1[1]+res2[1]
    
    total_words = len(words)
    guessed, attempts = parallel(total_words, guess, reduce)
    print(f"Guess percentage: {guessed/total_words*100:.2f}% - Attempts mean: {attempts/total_words:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test", action="store_true", help="Test the solver on the whole dictionary")
    args = parser.parse_args()

    scores = prepare("words")
    if args.test:
        test(list(scores.keys()))
    else:
        guess_word(scores)