from ast import keyword
import re
from utils import *
import spacy
from nltk import Tree
import json

nlp = spacy.load("en_core_web_sm")
debug = False


if debug:   
    dataset = ["Some people who are exposed to chronically stressful work conditions can experience job burnout , which is a general sense of emotional exhaustion and cynicism in relation to one \u2019s job ( Maslach & Jackson , 1981 ) ."]
else:
    dataset = [i[1] for i in dataset_convert('./train') if i[0]]

keyword_group_length = 5

patterns = {
    "after":".*(?:(?:is|are) referred to as|called) *([a-zA-Z0-9' ]+)(?:[,.;-]|to|and|or)"

}
def preprocessing_dataset(dataset):
    processed_dataset = []
    for each in dataset:
        if '(' in each:

            processed_dataset.append(re.sub(r'\([^)]*\)', '', each))
        else:
            new = re.sub(r'\u2019', '\'', each)
            #print("new",new)
            processed_dataset.append(new)




    return processed_dataset

def regex_match(patterns, doc):
    sentence = doc.text
    
    for each in patterns.keys():
        z = re.match(patterns[each], sentence)
        if debug:
   
            print("sentence",sentence)
            print("z",z)
        if z:

            result = z.groups()[0]
            if each == "after":
                breakdown_list = ["prep","advmod"]
                doc=nlp(result)
                sents_list = list(doc.sents)[0]
                for term_idx in range(len(sents_list)):
                    if sents_list[term_idx].dep_ in breakdown_list:
                        return sents_list[:term_idx].text



            if debug:
                print("result",result)

            # We want definition keyword length <= threhold, so convert to list first.
            result_split = result.strip().split(' ')
            if len(result_split) <= keyword_group_length:
                #print("return")
                return result







def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:

        return node.orth_


def nltk_tree_to_list(node, min_value, max_value):
    if node.n_lefts + node.n_rights > 0:
        tmp_min, tmp_max = None,None
        for child in node.children:

            if child.i<min_value:
                min_value = child.i
            elif child.i > max_value:
                max_value = child.i

            tmp_min, tmp_max = nltk_tree_to_list(child, min_value, max_value) 
            #print("tmp_min,max",tmp_min, tmp_max)
            #print("tmp min,max",min_value, max_value)
            if tmp_min < min_value:
                min_value = tmp_min
            elif tmp_max > max_value:
                max_value = tmp_max
        #print("cur range",min_value, max_value)
        return min_value,max_value
    else:
        if node.i<min_value:
            min_value = node.i
        elif node.i > max_value:
            max_value = node.i
        #print("cur leaf range",min_value, max_value)
        return min_value, max_value


def dependencies_pos_check(child):
    return child.dep_ in ["nsubj","dobj","nsubjpass"] or \
    (child.pos_ == "NOUN" and child.dep_ == "attr")

def processing_keywords(raw_keyword):
    remove_space = raw_keyword.strip()
    to_lower = remove_space.lower()
    split_list = to_lower.split(' ')

    length = len(split_list)

    if length == 1:
        return raw_keyword
    
    start_words = ["an","a"]

    for each in start_words:
        if each == split_list[0]:
            word = " ".join(split_list[1:])
            return word
        
    return to_lower

def extract_key_term_from_sentence(input_sentence):
    doc=nlp(input_sentence)
    if debug:
        print("------------")
        print("noun chunks")
        
        for chunk in doc.noun_chunks:
            print(chunk)

        print("------------")
    
    sentences = list(doc.sents)

    for sentence in sentences:
        #print(sentence.root)
        if debug:
            for each_term in sentence:
                print(each_term,each_term.dep_,each_term.pos_)

            [to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]


        #for child in sentence.root.children:
        #    print("child, label,dep",child,child.pos_,child.dep_ )
        raw_keyword = regex_match(patterns, doc)
        if debug:
            print("raw_keyword",raw_keyword)
        #print("raw_keyword",raw_keyword)
        #raw_keyword = False
        if not raw_keyword:
            for child in sentence.root.children:
                if dependencies_pos_check(child):

                    root_index = child.i
                    min_value = root_index
                    max_value = root_index
                    min_value, max_value = nltk_tree_to_list(child, min_value, max_value)
                    
                    #[to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]

                    range_length = max_value-min_value+1
                    if range_length <= keyword_group_length:
                        #result = doc[min_value:max_value+1]
                        #print("result",result)
                        raw_keyword = doc[min_value:max_value+1].text
                        keyword = processing_keywords(raw_keyword)
                        return keyword
                    

        else:
            keyword = processing_keywords(raw_keyword)
            #print("final",keyword)

            return keyword
                #break
processed_dataset = preprocessing_dataset(dataset)
individual_test = True      
if individual_test:
    #dataset = ["If this mechanism fails , multiple sperm can fuse with the egg , resulting in polyspermy ."]
    
    final_result = {}  
    all_len = len(processed_dataset)
    for each_index in range(len(processed_dataset)):
        if each_index%20==0 or each_index == all_len-1:
            print("processing %d/%d"%(each_index,all_len))
        tmp_result = extract_key_term_from_sentence(processed_dataset[each_index])
        #print("result keyword",tmp_result)
        if tmp_result:
            #print(tmp_result)
            #print(type(tmp_result))
            #assert 1==2
            final_result[tmp_result] = dataset[each_index]
            #final_result[dataset[each_index]] = 0
        
    #print(final_result)
    if debug:
        assert 1==2


    with open("train_extraction_test.json",'w') as obj:
        obj.write(json.dumps(final_result))



