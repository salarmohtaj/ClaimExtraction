from torchtext.data.metrics import bleu_score

s1 = "I love you so so very your dog"
output = [s1.split(),s1.split()[:4]]
s2 = "I love you so so very much"
truesentece = [s2.split(),s2.split()[:4]]

print(output)
print(truesentece)

print(bleu_score(truesentece, output))

candidate_corpus = [['My', 'full', 'pytorch', 'test'], ['Another', 'Sentence']]
references_corpus = [[['My', 'full', 'pytorch', 'test'], ['Completely', 'Different']], [['No', 'Match']]]
print(bleu_score(candidate_corpus, references_corpus))

print(candidate_corpus)
print(references_corpus)


import nltk
import nltk

hypothesis = ['It', 'is', 'a', 'cat', 'at', 'room']
reference = ['It', 'is', 'a', 'cat', 'inside', 'the', 'room']
#there may be several references
BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)
print(BLEUscore)


from rouge import Rouge

hypothesis = "the #### transcript is a written version of each day 's cnn student news program use this transcript to he    lp students with reading comprehension and vocabulary use the weekly newsquiz to test your knowledge of storie s you     saw on cnn student news"

reference = "this page includes the show transcript use the transcript to help students with reading comprehension and     vocabulary at the bottom of the page , comment for a chance to be mentioned on cnn student news . you must be a teac    her or a student age # # or older to request a mention on the cnn student news roll call . the weekly newsquiz tests     students ' knowledge of even ts in the news"

rouge = Rouge()
scores = rouge.get_scores(hypothesis, reference)
print(scores[0]["rouge-1"])

#hypothesis = ["I love you so so very your dog","a"]
hypothesis = ["a", "i c", "s"]
reference  = ["a", "i c", "s"]
#reference = ["I love you so so very much","b"]

rouge = Rouge()
scores = rouge.get_scores(hypothesis, reference)
print(scores[0]["rouge-1"])