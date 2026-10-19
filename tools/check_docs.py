"""Check built local links/images/downloads and paired-language targets."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit, unquote

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
    print('PASS:', len(pages), 'HTML pages; local links, images and downloads exist.')


if __name__ == '__main__':
    main()
