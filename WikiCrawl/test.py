from SPARQLWrapper import SPARQLWrapper, JSON

sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

sparql.setQuery("""
SELECT ?person ?personLabel
WHERE
{
  ?person wdt:P31 wd:Q5; # instance of any subclass of human
          wdt:P569 ?dob.
  FILTER("2015-01-01"^^xsd:dateTime <= ?dob && ?dob < "2016-01-01"^^xsd:dateTime).
  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE]". }
}
""")
sparql.setReturnFormat(JSON)
results = sparql.query().convert()
print(results)

'''
from wiki_dump_reader import Cleaner, iterate

cleaner = Cleaner()
for title, text in iterate('*wiki-*-pages-articles.xml'):
    text = cleaner.clean_text(text)
    cleaned_text, links = cleaner.build_links(text)'''