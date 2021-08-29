import xml.sax
import mwparserfromhell
import subprocess


class WikiXmlHandler(xml.sax.handler.ContentHandler):
    """Content handler for Wiki XML data using SAX"""

    def __init__(self):
        xml.sax.handler.ContentHandler.__init__(self)
        self._buffer = None
        self._values = {}
        self._current_tag = None
        self._pages = []
        self._article_count = 0

    def characters(self, content):
        """Characters between opening and closing tags"""
        if self._current_tag:
            self._buffer.append(content)

    def startElement(self, name, attrs):
        """Opening tag of element"""
        if name in ('title', 'text'):
            self._current_tag = name
            self._buffer = []

    def endElement(self, name):
        """Closing tag of element"""
        if name == self._current_tag:
            self._values[name] = ' '.join(self._buffer)
        if name == 'page':
            self._article_count += 1
            # Send the page to the process article function
            page = self.process_article(**self._values,
                                        template='Infobox person')
            # If article is a book append to the list of books
            if page:
                self._pages.append(page)

    def process_article(title, text, timestamp, template='Infobox person'):
        """Process a wikipedia article looking for template"""

        # Create a parsing object
        wikicode = mwparserfromhell.parse(text)

        # Search through templates for the template
        matches = wikicode.filter_templates(matches=template)

        if len(matches) >= 1:
            # Extract information from infobox
            content = wikicode.strip_code().strip()
            return title, content


# Object for handling xml
handler = WikiXmlHandler()
# Parsing object
parser = xml.sax.make_parser()
parser.setContentHandler(handler)
# Parse the entire file
count = 0
for line in subprocess.Popen(['bzcat'],
                             stdin=open(data_path),
                             stdout=subprocess.PIPE).stdout:
    try:
        parser.feed(line)
        count += 1
        if count >= 3:
            break
    except StopIteration:
        break
books = handler._pages
