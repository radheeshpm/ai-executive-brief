"""Create a source-constrained AI executive briefing and publish RSS 2.0.
Only public RSS headlines and snippets are sent to Gemini. Never add private data.
"""
import datetime as dt
import email.utils
import html
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import feedparser
import requests
from google import genai
from google.genai import types

TZ = ZoneInfo('Asia/Kolkata')
NOW = dt.datetime.now(dt.timezone.utc)
LOCAL_DATE = NOW.astimezone(TZ).date()
FEED_PATH = Path('rss.xml')
REPO = os.getenv('GITHUB_REPOSITORY', 'radheeshpm/ai-executive-brief')
OWNER, REPO_NAME = REPO.split('/', 1)
BASE = f'https://{OWNER}.github.io/{REPO_NAME}'
MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')

# Public RSS/Atom sources. Some sources may be temporarily unavailable; failures are logged.
SOURCES = {
    'OpenAI': 'https://openai.com/news/rss.xml',
    'Anthropic': 'https://www.anthropic.com/rss.xml',
    'Google AI': 'https://blog.google/technology/ai/rss/',
    'Google Cloud': 'https://cloud.google.com/blog/products/ai-machine-learning/rss/',
    'Microsoft AI': 'https://blogs.microsoft.com/ai/feed/',
    'AWS ML': 'https://aws.amazon.com/blogs/machine-learning/feed/',
    'AWS Security': 'https://aws.amazon.com/blogs/security/feed/',
    'The Hacker News': 'https://feeds.feedburner.com/TheHackersNews',
    'TechCrunch AI': 'https://techcrunch.com/category/artificial-intelligence/feed/',
    'Finextra': 'https://www.finextra.com/rss/headlines.aspx',
}

def clean(value, limit=500):
    value = re.sub(r'<[^>]*>', ' ', str(value or ''))
    return re.sub(r'\s+', ' ', html.unescape(value)).strip()[:limit]

def gather():
    candidates = []
    for source, url in SOURCES.items():
        try:
            response = requests.get(url, timeout=18, headers={'User-Agent': 'AIExecutiveBriefRSS/1.0'})
            response.raise_for_status()
            parsed = feedparser.parse(response.content)
            if not parsed.entries:
                print(f'WARN: no entries from {source}', file=sys.stderr)
            for entry in parsed.entries[:25]:
                link = entry.get('link', '').strip()
                if not link.startswith('https://') and not link.startswith('http://'):
                    continue
                date_tuple = entry.get('published_parsed') or entry.get('updated_parsed')
                if not date_tuple:
                    continue  # No verifiable publication date: do not infer one.
                import calendar
                published = dt.datetime.fromtimestamp(calendar.timegm(date_tuple), dt.timezone.utc)
                age_hours = (NOW - published).total_seconds() / 3600
                if not (-2 <= age_hours <= 96):
                    continue
                candidates.append({
                    'id': '', 'source': source, 'title': clean(entry.get('title'), 240),
                    'summary': clean(entry.get('summary', ''), 650),
                    'url': link, 'published': published.isoformat(), 'age_hours': round(age_hours, 1)
                })
        except Exception as exc:
            print(f'WARN: {source}: {exc}', file=sys.stderr)
    # Deduplicate by normalized URL, then by near-identical headline.
    seen_urls, seen_titles, articles = set(), set(), []
    for item in sorted(candidates, key=lambda x: x['published'], reverse=True):
        parsed = urlparse(item['url'])
        url_key = (parsed.netloc.lower(), parsed.path.rstrip('/').lower())
        title_key = re.sub(r'[^a-z0-9]+', '', item['title'].lower())[:100]
        if url_key in seen_urls or title_key in seen_titles or not title_key:
            continue
        seen_urls.add(url_key)
        seen_titles.add(title_key)
        item['id'] = f'S{len(articles)+1}'
        articles.append(item)
    return articles[:75]

def previous_urls():
    if not FEED_PATH.exists():
        return set()
    try:
        root = ET.parse(FEED_PATH).getroot()
        links = set()
        for item in root.findall('./channel/item')[:7]:
            for a in item.findall('./sourceLinks/link'):
                if a.text:
                    links.add(a.text.strip())
        return links
    except ET.ParseError:
        return set()

def generate(articles):
    prior = previous_urls()
    for article in articles:
        article['reported_before'] = article['url'] in prior
    prompt = f'''You are an enterprise AI intelligence editor. Today in India is {LOCAL_DATE.isoformat()}.
You have ONLY the source records in the JSON below. Do not use external knowledge to claim a fresh event.
Headlines and snippets are not independently verified facts; label source claims as "reported" or "announced".
Never invent publication dates, product versions, incidents, figures, links or quotes.
Choose 5-10 HIGH SIGNAL items if supported; fewer if evidence is thin. Prefer last 24h; older items (up to 96h) only when important and clearly date-stamped.
Avoid articles marked reported_before unless a material new update is evident.
Cover OpenAI, Anthropic, Google, Microsoft, AWS, AI security, enterprise, financial services, engineering, regulation when supported. Do NOT manufacture category coverage.
For each selected story, cite only one or more exact source IDs. Distinguish official vendor announcements from journalism and unverified allegations. Since only RSS snippets are available, say "according to [source]" for material claims; do not claim independent verification.
Business implications and recommended actions are your analysis, not factual announcements.
Return ONLY JSON object with keys: executive_summary (array of 2-4 strings), stories (array of objects each with headline, source_ids [array], what_happened, why_it_matters, banking_action, confidence), watch (array of 3 strings), experiment (string), bottom_line (string).
Use concise readable prose. Include confidence as "Official announcement", "Reported", or "Limited evidence".
Source records:\n{json.dumps(articles, ensure_ascii=False)}'''
    client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type='application/json', temperature=0.2),
    )
    data = json.loads(response.text)
    if not isinstance(data.get('stories'), list):
        raise ValueError('Gemini did not return stories array')
    lookup = {a['id']: a for a in articles}
    validated = []
    for story in data['stories'][:10]:
        ids = [sid for sid in story.get('source_ids', []) if sid in lookup]
        if not ids:
            continue
        story['source_ids'] = ids
        validated.append(story)
    if not validated:
        raise ValueError('No source-grounded stories returned; not publishing')
    data['stories'] = validated
    return data, lookup

def text_element(parent, tag, value):
    ET.SubElement(parent, tag).text = str(value)

def render(data, lookup):
    sections = [f'<h2>AI Executive Intelligence — {LOCAL_DATE:%d %B %Y}</h2>',
                '<p><em>Curated from dated public RSS items. Linked sources support the news claims; implications are editorial analysis. Headlines/snippets may not independently establish every claim.</em></p>',
                '<h3>60-second executive summary</h3><ul>']
    for x in data.get('executive_summary', [])[:4]:
        sections.append(f'<li>{html.escape(str(x))}</li>')
    sections.append('</ul><h3>Key developments</h3>')
    urls = []
    for idx, story in enumerate(data['stories'], 1):
        sections.append(f'<h4>{idx}. {html.escape(str(story.get("headline", "")))}</h4>')
        for label, key in [('What happened', 'what_happened'), ('Why it matters', 'why_it_matters'), ('Banking / delivery action', 'banking_action')]:
            sections.append(f'<p><strong>{label}:</strong> {html.escape(str(story.get(key, "")))}</p>')
        sections.append(f'<p><em>Evidence: {html.escape(str(story.get("confidence", "Reported")))}</em><br>Sources: ')
        refs = []
        for sid in story['source_ids']:
            src = lookup[sid]
            urls.append(src['url'])
            refs.append(f'<a href="{html.escape(src["url"], quote=True)}">{html.escape(src["source"])} — {html.escape(src["published"][:10])}</a>')
        sections.append(' | '.join(refs) + '</p>')
    sections.append('<h3>Three things to watch</h3><ol>')
    for x in data.get('watch', [])[:3]:
        sections.append(f'<li>{html.escape(str(x))}</li>')
    sections.append('</ol><h3>Experiment / proposal</h3>')
    sections.append(f'<p>{html.escape(str(data.get("experiment", "")))}</p>')
    sections.append('<h3>Bottom line</h3>')
    sections.append(f'<p>{html.escape(str(data.get("bottom_line", "")))}</p>')
    return '\n'.join(sections), list(dict.fromkeys(urls))

def publish(body, urls):
    if FEED_PATH.exists():
        root = ET.parse(FEED_PATH).getroot()
        channel = root.find('channel')
        if channel is None:
            raise ValueError('Existing RSS has no channel')
    else:
        root = ET.Element('rss', {'version': '2.0'})
        channel = ET.SubElement(root, 'channel')
    # Normalize channel metadata and replace placeholder URLs.
    for tag, val in [('title', 'AI Executive Intelligence Brief'), ('link', BASE + '/rss.xml'),
                     ('description', 'Daily sourced AI intelligence, enterprise and banking implications'),
                     ('language', 'en')]:
        elem = channel.find(tag)
        if elem is None:
            elem = ET.SubElement(channel, tag)
        elem.text = val
    guid = f'ai-executive-brief-{LOCAL_DATE.isoformat()}'
    for old in channel.findall('item'):
        if old.findtext('guid') == guid:
            print('Today\'s edition already exists; skipping duplicate publication')
            return
    item = ET.Element('item')
    text_element(item, 'title', f'AI Executive Intelligence — {LOCAL_DATE:%d %B %Y}')
    text_element(item, 'link', BASE + '/rss.xml')
    text_element(item, 'guid', guid)
    text_element(item, 'pubDate', email.utils.format_datetime(NOW))
    text_element(item, 'description', body)
    # Private bookkeeping in RSS: retained for deduplication; ignored by readers.
    source_links = ET.SubElement(item, 'sourceLinks')
    for url in urls:
        text_element(source_links, 'link', url)
    channel.insert(0, item)
    for old in channel.findall('item')[31:]:
        channel.remove(old)
    ET.indent(root, space='  ')
    ET.ElementTree(root).write(FEED_PATH, encoding='utf-8', xml_declaration=True)
    print(f'Published {guid} with {len(urls)} cited source links')

def main():
    if not os.getenv('GEMINI_API_KEY'):
        raise RuntimeError('GEMINI_API_KEY secret is missing')
    articles = gather()
    print(f'Collected {len(articles)} unique, dated articles (last 96h)')
    if len(articles) < 3:
        raise RuntimeError('Too few dated sources; refusing to publish an unreliable briefing')
    data, lookup = generate(articles)
    body, urls = render(data, lookup)
    publish(body, urls)

if __name__ == '__main__':
    main()
