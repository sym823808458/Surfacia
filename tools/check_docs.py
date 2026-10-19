"""Check built local links/images/downloads and paired-language targets."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit, unquote
import json
import re

ROOT = Path(__file__).resolve().parents[1] / 'docs/build/html'


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attrs):
        data = dict(attrs)
        if tag == 'a' and data.get('href'):
            self.urls.append(data['href'])
        if tag in ('img', 'script') and data.get('src'):
            self.urls.append(data['src'])
        if tag == 'link' and data.get('href'):
            self.urls.append(data['href'])


def main():
    failures = []
    pages = list(ROOT.rglob('*.html'))
    assert pages, 'Build docs first'
    for page in pages:
        parser = Links()
        parser.feed(page.read_text(encoding='utf-8'))
        for url in parser.urls:
            parsed = urlsplit(url)
            if parsed.scheme or parsed.netloc or not parsed.path or parsed.path.startswith('/'):
                continue
            target = (page.parent / unquote(parsed.path)).resolve()
            if target.is_dir():
                target /= 'index.html'
            if not target.is_file():
                failures.append(str(page.relative_to(ROOT)) + ': ' + url)
    assert not failures, '\n'.join(failures)
    retired = re.compile(r'\bSPES\b|Go\s+Beyond|SHAP[- ]cliff', re.I)
    for source in (ROOT.parents[1] / 'source').rglob('*.rst'):
        assert not retired.search(source.read_text(encoding='utf-8')), 'Retired topic in current manual: ' + str(source)
    raw_index = (ROOT / 'searchindex.js').read_text(encoding='utf-8')
    index = json.loads(raw_index.removeprefix('Search.setIndex(').removesuffix(')'))
    for field in ('terms', 'titleterms'):
        assert not {term: docs for term, docs in index[field].items() if docs and re.search(r'spes|shap.cliff', term, re.I)}, 'Retired topic remains searchable'
    print('PASS:', len(pages), 'HTML pages; local links, images and downloads exist.')
    print('PASS: current manual content and active search terms exclude retired ranking topics.')


if __name__ == '__main__':
    main()
