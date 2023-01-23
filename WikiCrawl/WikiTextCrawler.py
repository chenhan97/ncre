import requests
from requests import utils
from SPARQLWrapper import SPARQLWrapper, JSON

proxy = {
    "http": 'https://130.41.47.235:8080',
    "http": 'https://130.41.65.206:8080',
    "http": 'https://130.41.41.175:8080',
    "http": 'https://208.196.136.14:3128',
    "http": 'https://47.253.0.126:31775',
    "http": 'https://151.181.90.10:80',
    "http": 'https://198.49.68.80:80',
    "http": 'https://144.126.141.115:1010',
    "http": 'https://50.87.181.51:80',
    "http": 'https://154.83.29.202:999',
    "http": 'https://20.111.54.16:80',
    "http": 'https://154.83.29.204:999',
    "http": 'https://154.83.29.205:999',
    "http": 'https://154.83.29.200:999',
    "http": 'https://154.83.29.203:999',
    "http": 'https://154.83.29.201:999',
    "http": 'https://199.188.93.153:8000',
    "http": 'https://74.208.205.5:80',
    "http": 'https://157.245.222.183:80',
    "http": 'https://20.206.106.192:8123',
    "http": 'https://64.227.23.88:8118',
    "http": 'https://23.107.176.65:32180',
    "http": 'https://47.241.238.30:1081',
    "http": 'https://138.2.29.201:3128',
    "http": 'https://207.180.216.38:3130',
    "http": 'https://74.205.128.200:80',
    "http": 'https://149.28.155.188:22002',
    "http": 'https://104.45.128.122:80',
    "http": 'https://154.83.29.206:999',
    "http": 'https://205.201.49.132:53281',
    "http": 'https://158.101.142.112:3128',
    "http": 'https://65.108.9.181:80',
    "http": 'https://23.107.176.100:32180',
    "http": 'https://47.243.60.151:1081',
    "http": 'https://159.65.69.186:9300',
    "http": 'https://162.243.244.206:80',
    "http": 'https://162.19.50.37:80',
    "http": 'https://143.198.242.86:8048'}


def sqarql_query_get_people_id(timespan):
    """
    :return: all people's wikidata id, who was born in the defined time span
    """
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    query = "SELECT ?item WHERE {?item wdt:P31 wd:Q5 . ?item wdt:P569 ?birth . FILTER (?birth >= " \
            "\"" + timespan[0] + "\"^^xsd:dateTime) FILTER (?birth <= \"" + timespan[1] + "\"^^xsd:dateTime)}"
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    clean_results = []
    for item in results['results']['bindings']:
        wikidata_id = item['item']['value'].split("/")[-1]
        clean_results.append(wikidata_id)
    return clean_results


def get_wikipedia_url_from_wikidata_id(wikidata_id, lang='en'):
    url = (
        'https://www.wikidata.org/w/api.php'
        '?action=wbgetentities'
        '&props=labels|sitelinks/urls'
        f'&ids={wikidata_id}'
        '&format=json')
    json_response = requests.get(url).json()
    entities = json_response.get('entities')
    if entities:
        entity = entities.get(wikidata_id)
        if entity:
            sitelinks = entity.get('sitelinks')
            sitelink = sitelinks.get(f'{lang}wiki')
            if sitelink:
                wiki_url = sitelink.get('url')
                if wiki_url:
                    label = entity.get('labels').get(lang)
                    if label:
                        return label['value']
    return None


def get_wiki_content(name):
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) '
                             'Chrome/39.0.2171.95 Safari/537.36'}
    response = requests.get('https://en.wikipedia.org/w/api.php', params={
        'action': 'query',
        'format': 'json',
        'titles': name,
        'prop': 'extracts',
        'explaintext': True,
        'exsectionformat': 'plain'}, headers=headers, proxies=proxy).json()
    page = next(iter(response['query']['pages'].values()))
    if 'extract' in page.keys():
        return page['extract']
    else:
        return None
