from pathlib import Path
import re
ROOT = Path(__file__).resolve().parents[2]
project = 'Surfacia'
author = 'Yuming Su'
copyright = '2026, Yuming Su'
release = re.search(r'^version = "([^"]+)"', (ROOT/'pyproject.toml').read_text(encoding='utf-8'), re.M).group(1)
version = release
language = 'en'
extensions = ['sphinx.ext.mathjax', 'sphinx_copybutton', 'sphinx_design']
templates_path = ['_templates']
exclude_patterns = []
html_theme = 'furo'
html_title = 'Surfacia documentation'
html_logo = '_static/images/surfacia_logo.png'
html_favicon = '_static/favicon.ico'
html_static_path = ['_static']
html_css_files = ['css/custom.css']
html_theme_options = {
    'sidebar_hide_name': True,
    'source_repository': 'https://github.com/sym823808458/Surfacia',
    'source_branch': 'main', 'source_directory': 'docs/source/',
    'light_css_variables': {'color-brand-primary': '#175b70', 'color-brand-content': '#175b70'},
    'dark_css_variables': {'color-brand-primary': '#7fc9c9', 'color-brand-content': '#7fc9c9'},
}
copybutton_prompt_text = r'>>> |\.\.\. |\$ '
copybutton_prompt_is_regexp = True

def language_context(app, pagename, templatename, context, doctree):
    is_chinese = pagename.startswith('zh/')
    other = pagename[3:] if is_chinese else 'zh/' + pagename
    if other not in app.env.found_docs:
        other = 'index' if is_chinese else 'zh/index'
    context['other_language_page'] = other
    context['other_language_label'] = 'English' if is_chinese else '中文'
    context['current_language_home'] = 'zh/index' if is_chinese else 'index'
    context['language'] = 'zh-CN' if is_chinese else 'en'
    routes = ['getting_started/installation', 'getting_started/quick_start',
              'getting_started/basic_concepts', 'user_guide/workflows',
              'descriptors/index', 'user_guide/results', 'examples/index',
              'commands/index', 'api/index', 'integrations/index',
              'user_guide/best_practices', 'user_guide/troubleshooting',
              'migration', 'citation', 'changelog', 'contributing', 'license']
    prefix = 'zh/' if is_chinese else ''
    context['manual_navigation'] = [(prefix + route, app.env.titles[prefix + route].astext()) for route in routes]

def setup(app):
    app.connect('html-page-context', language_context)
