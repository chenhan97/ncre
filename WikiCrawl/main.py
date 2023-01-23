import time
import WikiTextCrawler
import WikidataCrawlerSingle
import spacy
import jsonlines

nlp = spacy.load('en_core_web_trf')

import os
os.environ['HTTP_PROXY'] = 'https://130.41.47.235:8080'

count = 5141
whole_data_list = []
for time_ in range(1913, 2000):
    time_start = str(time_) + "-01-01"
    time_end = str(time_) + "-12-31"
    timespan = [time_start, time_end]
    list_of_people_id = WikiTextCrawler.sqarql_query_get_people_id(timespan)
    shorted_list_of_people_id = []
    #for id in list_of_people_id:
    #    name = WikiTextCrawler.get_wikipedia_url_from_wikidata_id(id)
    #    if name:
    #        shorted_list_of_people_id.append(id)
    num_batch = int(len(list_of_people_id) / 20)
    for batch_count in range(num_batch):
        wikidata_results = WikidataCrawlerSingle.sqarql_query(list_of_people_id[batch_count*20: (batch_count+1)*20])
        formatted_wikidata_results = WikidataCrawlerSingle.query_result_converter(wikidata_results)
        # clean wikidata facts
        for person in formatted_wikidata_results.keys():
            facts = WikidataCrawlerSingle.clean_facts(formatted_wikidata_results[person])
            complete_fact_flag = False
            for fact in facts:
                if len(fact['qualifiers'])!=0:
                    complete_fact_flag = True
                    break
            if complete_fact_flag:
                time.sleep(2)
                plain_text = WikiTextCrawler.get_wiki_content(person)
                if plain_text:
                    print(person)
                    paragraph_split = plain_text.split("\n")
                    sentences = [i.text for para in paragraph_split for i in nlp(para).sents]
                    labeled_data = {'doc_id': count, 'subject': person, 'data': facts}
                    valid_data_flag = False
                    for i in range(len(facts)):
                        fact = facts[i]
                        main_sent_list = []
                        supp_sent_list = []
                        for sent in sentences:
                            if person in sent and fact['object'] in sent:
                                main_sent_list.append(sent)
                        if len(fact['qualifiers']) != 0:
                            for qualifier in fact['qualifiers']:
                                for sent in sentences:
                                    if qualifier in sent:
                                        supp_sent_list.append(sent)
                        if len(main_sent_list) != 0 and len(supp_sent_list) != 0:
                            labeled_data['data'][i]['main_sent'] = main_sent_list
                            labeled_data['data'][i]['supp_sent'] = supp_sent_list
                            valid_data_flag = True
                    if valid_data_flag:
                        with open("dataset/doc/" + str(count), mode='w', encoding='utf=8') as writer:
                            for sent in sentences:
                                writer.write(sent + "\n")
                        #whole_data_list.append(labeled_data)
                        if count == 0:
                            with jsonlines.open('dataset/alldata.json', mode='w') as writer:
                                writer.write(labeled_data)
                        else:
                            with jsonlines.open('dataset/alldata.json', mode='a') as writer:
                                writer.write(labeled_data)
                        count += 1
