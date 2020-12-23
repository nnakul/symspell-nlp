# ABOUT
The project realises a feauture of correcting invalid words in English language using Symmetric Delete Spelling Correction Algorithm. Spelling suggestion is a feature of many computer software applications used to suggest plausible replacements for words that are likely to have been misspelled. This feauture is commonly included in Internet search engines, word processors and spell checkers. 

# SYMMETRIC DELETE SPELLING CORRECTION ALGORITHM
Any spell checker must have some data about the words in the target language, either in general usage or with specialized knowledge (like medical vocabulary). This can come from:<ul>
<li>A dictionary of all known words.
<li>A text corpus which includes typical text, known to be correctly spelled.
<li>A list of frequently misspelled words, mapping errors to corrections.
</ul>
In this project, only the first two sources of data are used. List of frequently misspelled words is not maintained to reduce any additional look-up instructions.
However such a list, possibly including multi-word phrases, can simply be consulted to see if any of the input words or phrases are present in the list. This functionality can be added depending on the requirements and the nature of input queries.<br>

To make use of a dictionary without a pre-existing mapping from misspellings to corrections, the typical technique is to calculate the edit distance between an input word and any given word in the dictionary. The Damerau–Levenshtein distance metric considers an "edit" to be the insertion, deletion, transposition or substitution (with another letter) of one letter. Dictionary words that are an edit distance of 1 away from the input word are considered highly likely as corrections, edit distance 2 less likely, and edit distance 3 sometimes included in suggestions and sometimes ignored.<br>

Because a dictionary of known words is very large, calculating the edit distance between an input word and every word in the dictionary is computationally intensive and thus relatively slow. For a more efficient and practical application, the following can be some of the solutions:<br>
&nbsp;&nbsp;o&nbsp;&nbsp;Various data structures can be utilized to speed up storage lookups, such as BK-trees.<br>
&nbsp;&nbsp;o&nbsp;&nbsp;A faster approach can be used to generate all the permutations from an input word of all possible edits. For a word of length n and an alphabet of size a, for edit distance 1 there are at most n deletions, n-1 transpositions, a*n alterations, and a*(n+1) insertions.[5] Using only the 26 letters in the English alphabet, this would produce only 54*n+25 dictionary lookups, minus any duplicates (which depends on the specific letters in the word). This is relatively small when compared to a dictionary of hundreds of thousands of words. However, tens or hundreds of thousands of lookups might be required for edit distance 2 and greater.<br>

A further innovation known as SymSpell ("sym" as in "symmetry") can be adopted to speed up the input-time calculation by utilizing the fact that only permutations involving deletions need to be generated for input words if the same deletion permutations are per-calculated on the dictionary.

The algorithms described so far do not deal well with correct words that are not in the dictionary. Common sources of unknown words in English are compound words and inflections, such as -s and -ing.[4] These can be accommodated algorithmically, especially if the dictionary contains the part of speech.





These algorithms have also assumed that all errors of a given distance are equally probable, which is not true. Errors that involve spelling phonetically where English orthography is not phonetic are common, as are errors that repeat the same letter or confuse neighboring letters on a QWERTY keyboard. If a large set of known spelling errors and corrections is available, this data can be used to generate frequency tables for letter pairs and edit types; these can be used to more accurately rank suggestions.[4] It is also more common than chance for a word to be spelled in the wrong dialect compared to the rest of the text, for example due to American and British English spelling differences.[4]

Spelling suggestions can also be made more accurate by taking into account more than one word at a time.[4] Multi-word sequences are known as n-grams (where n is the number of words in the sequence). A very large database of n-grams up to 5 words in length is available from Google for this and other purposes.[6]

Others have experimented with using large amounts of data and deep learning techniques (a form of machine learning to train neural networks to perform spelling correction.
