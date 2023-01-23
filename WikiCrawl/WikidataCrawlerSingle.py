from SPARQLWrapper import SPARQLWrapper, JSON
import re


def sqarql_query(qs):
    """
    :param qs: qs is the Q index of a subject. E.G., the input can be Q7186 (that's marie curie)
    :return: all properties and qualifiers related to marie curie
    """
    qs_string = ""
    for i in qs:
        qs_string = qs_string + "wd:"+str(i)+" "
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql", agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36')
    query = "SELECT ?personLabel ?wdLabel ?ps_Label ?wdpqLabel ?pq_Label WHERE { VALUES ?person {"+qs_string+"}  " \
            "?person ?p ?statement .  ?statement ?ps ?ps_ .?wd wikibase:claim ?p. ?wd wikibase:statementProperty ?ps. " \
            "OPTIONAL {?statement ?pq ?pq_ .?wdpq wikibase:qualifier ?pq .}SERVICE wikibase:label { bd:serviceParam " \
            "wikibase:language \"en\" } } ORDER BY ?wd ?statement ?ps_ "
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return results


'''
def query_result_converter(query_result):
    """
    :param query_result: the query result from function sqarql_query
    :return: a list of fact dictionary: format:[{'relation': " ", 'object': " ","sub_relations": [], "qualifiers": []},...,{}]
    """
    current_fact = {'relation': " ", 'object': " "}
    all_facts, sub_rel_list, qualifiers = [], [], []
    for fact_dict in query_result['results']['bindings']:
        rel_label = fact_dict['wdLabel']['value']
        object_label = fact_dict['ps_Label']['value']
        person_label = fact_dict['person']['value'].split("/")[-1]
        if rel_label == current_fact['relation'] and object_label == current_fact['object']:
            if "wdpqLabel" in fact_dict.keys():
                current_fact["sub_relations"].append(fact_dict['wdpqLabel']['value'])
                current_fact["qualifiers"].append(fact_dict['pq_Label']['value'])
        else:
            all_facts.append(current_fact)
            current_fact = {'relation': rel_label, 'object': object_label, "sub_relations": [], "qualifiers": []}
            if "wdpqLabel" in fact_dict.keys():
                current_fact["sub_relations"].append(fact_dict['wdpqLabel']['value'])
                current_fact["qualifiers"].append(fact_dict['pq_Label']['value'])
    return all_facts[1:]
'''


def query_result_converter(query_result):
    """
    :param query_result: the query result from function sqarql_query
    :return: a list of fact dictionary: format:[{'relation': " ", 'object': " ","sub_relations": [], "qualifiers": []},...,{}]
    """
    facts_of_all_persons = {}
    for fact_dict in query_result['results']['bindings']:
        if fact_dict['personLabel']['value'] not in facts_of_all_persons.keys():
            facts_of_all_persons[fact_dict['personLabel']['value']] = []
    for person in facts_of_all_persons.keys():
        current_fact = {'relation': " ", 'object': " "}
        all_facts, sub_rel_list, qualifiers = [], [], []
        for fact_dict in query_result['results']['bindings']:
            person_label = fact_dict['personLabel']['value']
            if person_label == person:
                rel_label = fact_dict['wdLabel']['value']
                object_label = fact_dict['ps_Label']['value']
                if rel_label == current_fact['relation'] and object_label == current_fact['object']:
                    if "wdpqLabel" in fact_dict.keys():
                        current_fact["sub_relations"].append(fact_dict['wdpqLabel']['value'])
                        current_fact["qualifiers"].append(fact_dict['pq_Label']['value'])
                else:
                    all_facts.append(current_fact)
                    current_fact = {'relation': rel_label, 'object': object_label, "sub_relations": [],
                                    "qualifiers": []}
                    if "wdpqLabel" in fact_dict.keys():
                        current_fact["sub_relations"].append(fact_dict['wdpqLabel']['value'])
                        current_fact["qualifiers"].append(fact_dict['pq_Label']['value'])
        facts_of_all_persons[person] = all_facts[1:]
    return facts_of_all_persons


def clean_facts(facts_list):
    """
    :param facts_list: the list from the output of function query_result_converter
    :return: a list of facts where none of the element has non-english text symbols
    TODO: the reason is most of them are urls and times
    """
    # remove all that include non-text symbols
    pattern = re.compile(r'^\s*[A-Za-z]+(?:\s+[A-Za-z]+)*\s*$')
    new_facts_list = []
    for fact in facts_list:
        objects = fact['object']
        if bool(pattern.match(objects)):
            if len(fact['sub_relations']) != 0:
                remaining_item_idx = []
                count = 0
                for qualifier in fact['qualifiers']:
                    if bool(pattern.match(qualifier)):
                        remaining_item_idx.append(count)
                    count += 1
                new_sub_relations = []
                new_qualifiers = []
                for i in remaining_item_idx:
                    new_sub_relations.append(fact['sub_relations'][i])
                    new_qualifiers.append(fact['qualifiers'][i])
                new_facts_list.append(
                    {'relation': fact['relation'], 'object': fact['object'], 'sub_relations': new_sub_relations,
                     'qualifiers': new_qualifiers})
            else:
                new_facts_list.append(fact)
    return new_facts_list
