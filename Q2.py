"""
Created on Sat May  6 10:48:40 2023
@author: Farzan Soleymani
"""
import nltk
from typing import Sequence

def find_word_combos_with_pronunciation(phonemes: Sequence[str]) -> Sequence[Sequence[str]]:
    nltk.download('cmudict')
    cmu_dict = nltk.corpus.cmudict.dict()
    phoneme_seq1 = []
    phoneme_seq2 = []
    for i in range(len(phoneme_in)):
        phoneme_seq1.append(phonemes[:i+1])
        phoneme_seq2.append(phonemes[i+1:])

    Possible_Words1 = []
    Possible_Words2 = []
    for wrods, pronunciations in cmu_dict.items():
        for phonemes in pronunciations:
            phoneme = [p.strip('012') for p in phonemes]
            for p_seq in phoneme_seq1:
                if phoneme == p_seq:
                    Possible_Words1.append(wrods)
            for p_seq in phoneme_seq2:
                if phoneme == p_seq:
                    Possible_Words2.append(wrods)
                
    Possible_Words = [Possible_Words1, Possible_Words2]

    return print(f"\n Found words withe given a sequence of phonemes: \n\n {Possible_Words}")

""" TEST: 
    In this code the stress levels (0-1-2) are removed from Phonemes.
    Given a sequence of phonemes, two sequences of possible words are generated,
    from all possible combinations of phoneme sequence. 
"""
if __name__ == '__main__':
    phoneme_in = ["DH", "EH", "R", "DH", "EH", "R"]
    words = find_word_combos_with_pronunciation(phoneme_in)

