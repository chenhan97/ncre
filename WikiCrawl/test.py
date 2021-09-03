from SPARQLWrapper import SPARQLWrapper, JSON
import requests
from bs4 import BeautifulSoup
import jsonlines
import json


def get_wikiid_of_title(title):
    response = requests.get(url="https://en.wikipedia.org/wiki/" + title, )
    soup = BeautifulSoup(response.content, 'html.parser')
    wikidata_link = soup.find(id="t-wikibase")
    cleaned_wikidata_link = wikidata_link.find_all('a', href=True)[0]['href']
    title_id = str(cleaned_wikidata_link).split("/")[-1]
    return title_id


def query_predicates_and_values(title_id):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    query = "SELECT ?wdLabel ?ps_Label ?wdpqLabel ?pq_Label { wd:" + title_id + "?p ?statement . ?statement ?ps ?ps_ . " \
                                                                                "?wd wikibase:claim ?p. ?wd " \
                                                                                "wikibase:statementProperty ?ps. OPTIONAL " \
                                                                                "{ ?statement ?pq ?pq_ . ?wdpq " \
                                                                                "wikibase:qualifier ?pq .}SERVICE " \
                                                                                "wikibase:label { bd:serviceParam " \
                                                                                "wikibase:language 'en' }} "
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    raw_results = sparql.query().convert()
    raw_results = raw_results['results']['bindings']
    results = {'title': None, 'data': []}
    qualifier_flag = 0
    for dic in raw_results:
        predicate = dic['wdLabel']['value']
        value = dic['ps_Label']['value']
        # how to combine the dictionaries that have the same objective and predicate
        if 'wdpqLabel' in dic.keys():
            qualifier_flag = 1
            qualifier = dic['pq_Label']['value']
            sub_pred = dic['wdpqLabel']['value']
            simplified_dict = {'relation': predicate, 'objective': value, 'sub-relation': sub_pred,
                               'sup-ent': qualifier}
            if len(old_dict) != 0 and predicate == old_dict['relation'] and value == old_dict['objective']:
                old_dict = old_dict.copy()
                old_dict.update(simplified_dict)
            else:
                old_dict = simplified_dict
        else:
            if qualifier_flag == 1:
                results['data'].append(old_dict)
            qualifier_flag = 2
            simplified_dict = {'relation': predicate, 'objective': value}
            results['data'].append(simplified_dict)
    return results


def extract_wikidata_to_json(text_path, json_path):
    title_names = []
    with open(text_path, 'rt') as fin:
        for l in fin.readlines():
            title_names.append("_".join(json.loads(l)[0].split()))
    title_ids = []
    count = 0
    for title in title_names:
        count += 1
        if count >= 10:
            break
        title_ids.append(get_wikiid_of_title(title))
    with jsonlines.open(json_path, mode='w') as writer:
        for i in range(len(title_ids)):
            print(title_names[i], i, title_ids[i])
            one_query_result = query_predicates_and_values(title_ids[i])
            one_query_result['title'] = title_names[i]
            writer.write(one_query_result)


extract_wikidata_to_json("pages.ndjson", 'output.json')
