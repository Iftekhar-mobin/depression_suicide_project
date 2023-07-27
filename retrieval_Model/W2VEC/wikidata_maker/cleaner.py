import re
import sys

def single_character_remover(text):
    collector = []
    text = re.sub(r'\W+', ' ', text)
    for items in text.split():
        if len(items) < 2:
            replaced = re.sub(r'[ぁ-んァ-ン]', '', items)
            replaced = re.sub(r'[A-Za-z]', '', replaced)
            replaced = re.sub(r'[0-9]', '', replaced)
            collector.append(replaced)
        else:
            collector.append(items)

    return ' '.join([temp.strip(' ') for temp in collector]) + '\n'

#with open('jp_wiki_processed_cleaned.txt', 'rt') as infile, open('cleaned_fresh.txt', "w") as outfile:  
with open('mhlw_s.txt', 'rt') as infile, open('cleaned_fresh_mhlw.txt', "w") as outfile:
    for line in infile:
         line = single_character_remover(line)
         outfile.write(line)

