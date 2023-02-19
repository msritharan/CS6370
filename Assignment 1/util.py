# Add your import statements here
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from sentenceSegmentation import *
from tokenization import *
from stopwordRemoval import *
from inflectionReduction import *

# Add any utility functions here
text = "He stared out the window at the snowy field. He'd been stuck in the house for close to a month and his only view of the outside world was through the window. There wasn't much to see. It was mostly just the field with an occasional bird or small animal who ventured into the field. As he continued to stare out the window, he wondered how much longer he'd be shackled to the steel bar inside the house. Was it enough? That was the question he kept asking himself. Was being satisfied enough? He looked around him at everyone yearning to just be satisfied in their daily life and he had reached that goal. He knew that he was satisfied and he also knew it wasn't going to be enough. The time to take action was now. All three men knew in their hearts this was the case, yet none of them moved a muscle to try. They were all watching and waiting for one of the others to make the first move so they could follow a step or two behind and help. The situation demanded a leader and all three men were followers."
sentence_segmented_text = SentenceSegmentation().punkt(text)
print(sentence_segmented_text)

tokenized_text = Tokenization().pennTreeBank(sentence_segmented_text)
print(tokenized_text)

stopword_removed_text = StopwordRemoval().fromList(tokenized_text)
print(stopword_removed_text)

lemmetized_text = InflectionReduction().reduce(stopword_removed_text)
print(lemmetized_text)
