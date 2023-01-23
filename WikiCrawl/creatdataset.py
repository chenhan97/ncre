import jsonlines
import itertools


def contained_entity_set(sent, entities):
    list_ent = [ent for ent in entities if ent in sent]
    return set(list_ent)


def exract_all_supp_combine(supp_sent_list, entities):
    all_supp_combine = []
    all_possible_combine = []
    count = 0
    if len(supp_sent_list) > 1:
        if len(supp_sent_list) > 10:
            search_range = 3
        else:
            search_range = len(supp_sent_list)
        for i in range(1, search_range + 1):
            temp = list(itertools.combinations(supp_sent_list, i))
            all_possible_combine.extend(temp)
    for comb in all_possible_combine:
        test_sent = list(comb)
        all_set = [contained_entity_set(sent, entities) for sent in test_sent]
        flag = True
        for i in range(len(all_set)):
            if i != len(all_set):
                new_list = all_set[:i] + all_set[i + 1:]
            else:
                new_list = all_set[:-1]
            for j in new_list:
                if all_set[i].issubset(j):
                    flag = False
                    break
        if flag:
            long_sent = " ".join(test_sent)
            count_ent = [long_sent.count(ent) for ent in entities]
            if 0 not in count_ent:
                all_supp_combine.append(test_sent)
                count += 1
        if count > 3:
            break
    return all_supp_combine


'''
# form the "complete" dataset 
count_ = 0
with jsonlines.open('dataset/cleandata.json') as reader, open('dataset/complete', mode='w', encoding='utf-8') as writer:
    for obj in reader:
        count_ += 1
        print(count_)
        head_entity = obj['subject']
        doc_id = obj["doc_id"]
        for instance in obj['data']:
            relation = instance['relation']
            tail_entity = instance['object']
            other_entities = instance["qualifiers"]
            all_entites = [_ for _ in other_entities] + [head_entity, tail_entity]
            supp_sent_list = instance['supp_sent']
            main_sent_list = instance['main_sent']
            new_supp_sent_list = [sent.replace("\n", " ") for sent in supp_sent_list if sent not in main_sent_list ]
            if len(new_supp_sent_list)>1:
                all_supp_combine = exract_all_supp_combine(new_supp_sent_list, other_entities)
            elif len(new_supp_sent_list) ==1:
                all_supp_combine = [new_supp_sent_list]
            else:
                continue
            count_samples = 0
            for main_sent in instance['main_sent']:
                new_main_sent = main_sent.replace("\n"," ")
                new_sent_list = [new_main_sent]
                for supp_comb in all_supp_combine:
                    new_sent_list.extend(supp_comb)
                    inputs = " ".join(new_sent_list)
                    input_other_ent = "|||".join(other_entities)
                    # add doc_id
                    count_ent = [inputs.count(ent) for ent in all_entites]
                    if 0 not in count_ent:
                        writer.write(inputs+"\t"+relation+"\t"+head_entity+"\t"+tail_entity+"\t"+input_other_ent+"\t"+"complete"+"\t"+str(doc_id)+"\n")
                        count_samples += 1
                    new_sent_list = []
                    #count_samples += 1
                    if relation == "given name" or relation == "family name":
                        if count_samples > 1:
                            break
                    elif count_samples>3:
                        break
                if relation == "given name" or relation == "family name":
                    if count_samples > 1:
                        break
                elif count_samples > 3:
                    break'''


'''
# form the over-complete dataset
import spacy
nlp = spacy.load('en_core_web_trf')
import random

with open('dataset/complete', encoding='utf-8') as reader, open('dataset/over-complete', encoding='utf-8', mode="w") as writer:
    for obj in reader:
        inputs, relation, head_entity, tail_entity, input_other_ent, complete, id = obj.split("\t")
        id = int(id.replace("\n",""))
        print(id)
        sent_list = []
        with open("dataset/doc/" + str(id), encoding='utf-8') as doc_reader:
            for sent in doc_reader:
                sent_list.append(sent.replace("\n", ""))
        sentences = [i.text for i in nlp(inputs).sents]
        new_main_sent = sentences[0]
        if new_main_sent in sent_list:
            for sent in sentences:
                if sent in sent_list:
                    sent_list.remove(sent)
        if len(sent_list) > 3:
            c1, c2, c3 = random.sample(sent_list, 3)
            append_sents = " ".join([c1, c2, c3])
        else:
            append_sents = " ".join(sent_list)
        writer.write(
            inputs + append_sents + "\t" + relation + "\t" + head_entity + "\t" + tail_entity + "\t" + input_other_ent + "\t" + "over-complete" + "\n")'''

'''
#  form the "None" dataset
import spacy
nlp = spacy.load('en_core_web_trf')
import random

with open('dataset/complete', encoding='utf-8') as reader, open('dataset/none', encoding='utf-8', mode='w') as writer:
    count = 0
    for obj in reader:
        instances = obj.split("\t")
        if len(instances) == 7:
            inputs, relation, head_entity, tail_entity, input_other_ent, complete, id = instances
        else:
            continue
        id = int(id.replace("\n", ""))
        count += 1
        print(count)
        sent_list = []
        with open("dataset/doc/" + str(id), encoding='utf-8') as doc_reader:
            for sent in doc_reader:
                sent_list.append(sent.replace("\n", ""))
        sentences = [i.text for i in nlp(inputs).sents]
        new_main_sent = sentences[0]
        if new_main_sent in sent_list:
            for sent in sentences:
                if sent in sent_list:
                    sent_list.remove(sent)
        if len(sent_list) > 5:
            c1, c2, c3,c4 = random.sample(sent_list, 4)
            append_sents = " ".join([c1, c2, c3,c4])
        else:
            append_sents = " ".join(sent_list)
        writer.write(append_sents + "\t" + relation + "\t" + head_entity + "\t" + tail_entity + "\t" + input_other_ent + "\t" + "none" + "\n")'''

'''
# form in-complete dataset
import spacy

nlp = spacy.load('en_core_web_trf')
import random


def delete_rand_items(items, n):
    to_delete = set(random.sample(range(len(items)), n))
    return [x for i, x in enumerate(items) if not i in to_delete]


with open('dataset/complete', encoding='utf-8') as reader, open('dataset/in-complete', encoding='utf-8',
                                                                mode='w') as writer:
    count = 0
    for obj in reader:
        instances = obj.split("\t")
        if len(instances) == 7:
            inputs, relation, head_entity, tail_entity, input_other_ent, complete, id = instances
        else:
            continue
        id = int(id.replace("\n", ""))
        count += 1
        print(count)
        sent_list = []
        with open("dataset/doc/" + str(id), encoding='utf-8') as doc_reader:
            for sent in doc_reader:
                sent_list.append(sent.replace("\n", ""))
        sentences = [i.text for i in nlp(inputs).sents]
        if len(sentences[1:]) > 2:
            n = 2
            input_sentence_list = delete_rand_items(sentences[1:], n)
        elif len(sentences[1:]) != 0:
            n = 1
            input_sentence_list = delete_rand_items(sentences[1:], n)
        else:
            input_sentence_list = []
        if len(input_sentence_list) != 0:
            writer.write(" ".join(sentences[
                                  1:]) + "\t" + relation + "\t" + head_entity + "\t" + tail_entity + "\t" + input_other_ent + "\t" + "in-complete" + "\n")
            new_main_sent = sentences[0]
            if new_main_sent in sent_list:
                for sent in sentences:
                    if sent in sent_list:
                        sent_list.remove(sent)
            if len(sent_list) > 2:
                c1, c2 = random.sample(sent_list, 2)
                append_sents = [c1, c2]
            else:
                append_sents = sent_list
            input_sentence_list.extend(append_sents)
            writer.write(" ".join(
                input_sentence_list) + "\t" + relation + "\t" + head_entity + "\t" + tail_entity + "\t" + input_other_ent + "\t" + "in-complete" + "\n")'''


#combine dataset together
count = 0
with open("dataset/complete", encoding='utf-8') as reader1, open("dataset/in-complete", encoding='utf-8') as reader2,open("dataset/over-complete", encoding='utf-8') as reader3,\
open("dataset/none", encoding='utf-8') as reader4,open("dataset/alldata", mode='w', encoding='utf-8') as writer5:
    count += 1
    for obj in reader1:
        instances = obj.split("\t")
        if len(instances) == 7:
            inputs, relation, head_entity, tail_entity, input_other_ent, complete, id = instances
            count += 1
            writer5.write(
                inputs + "\t" + relation + "\t" + head_entity + "\t" + tail_entity + "\t" + input_other_ent + "\t" + "complete" + "\n")

            print(count)
    for obj in reader2:
        count += 1
        writer5.write(obj)
        print(count)
    for obj in reader3:
        count += 1
        writer5.write(obj)
        print(count)
    for obj in reader4:
        count += 1
        writer5.write(obj)
        print(count)