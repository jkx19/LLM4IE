import stanza

stanford_parser = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency',
                                  use_gpu=False, download_method=None)

sentence = "Officials believe that this has left a huge loophole in which illegal drug traffickers operate."

tree = stanford_parser(sentence).sentences[0].constituency
print(tree)