"""
util.py - Internal Utility Module for Hawaiian-English Book Exploration
Author: Joe Winkie

This module provides helper functions and the `InternalState` class for working with
Hawaiian-English book data.

This is used for interactive data exploration notebooks and backend NLP components.
"""

import json
from collections import Counter, defaultdict

from bs4 import BeautifulSoup
from tokenizers.pre_tokenizers import Punctuation, Sequence, Whitespace
import os


def normalize_okina(text):
    """
    Replace left single quotation marks with the Hawaiian ʻokina character.
    """
    return text.replace(chr(8216), chr(699))


def clean_html_text(text):
    """
    Strips HTML tags and returns clean text.
    """
    soup = BeautifulSoup(f"<text>{text}</text>", "html.parser")
    return soup.get_text().strip()


class InternalState:
    """
    Encapsulates runtime state for Hawaiian-English book exploration, including:
    - Config loading
    - Book and page management
    - Tokenization and dictionary-based lookup
    """
    def __init__(self):
        self.english_to_hawaiian = defaultdict(list)
        self.hawaiian_to_english = defaultdict(list)

        # Diacritic replacements
        self.replacements = {
            "ʻ": "",
            "Ā": "A", "Ē": "E", "Ī": "I", "Ō": "O", "Ū": "U",
            "ā": "a", "ē": "e", "ī": "i", "ō": "o", "ū": "u",
        }

        self.loaded_pages = {}  # in-memory page cache
        self.pages: dict[str, list[dict[str, str]]] = {}

        # Load config
        with open("./config.json", "r") as f:
            self.config = json.load(f)

        # Load book list from JSONL
        self.books = []
        self.book_map = {}
        with open(self.config['booklist']) as f:
            for line in f.readlines():
                book = json.loads(line)
                book['img'] = f"/cover_img/{book['id']}.jpg"
                self.books.append(book)
                self.book_map[book['id']] = book

        # Tokenizer setup (whitespace + punctuation)
        self.pretokenizer = Sequence([Whitespace(), Punctuation()])

        # Load E2H dictionary from TSV
        self.load_f2e()

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenizes input text using whitespace and punctuation.
        """
        return [token for token, _ in self.pretokenizer.pre_tokenize_str(text)]

    def load_f2e(self):
        """
        Loads English-to-Hawaiian mappings from a TSV file and builds
        a reverse Hawaiian-to-English dictionary (diacritic-insensitive).
        """
        dic = {}
        with open(self.config["trussel_path"]) as fin:
            for line in fin:
                eng, *haws = line.strip().split("\t")
                for haw in haws:
                    haw = normalize_okina(haw)
                    if all(c.isupper() for c in haw):
                        continue
                    if haw not in dic:
                        dic[haw] = Counter()
                    dic[haw][eng] += 1

        # Build sorted list of translations
        newdic = {}
        for haw in dic:
            engs = [eng for eng, _ in dic[haw].most_common()]
            newdic[haw] = ", ".join(engs)

        # Save as diacritic-insensitive lookup
        self.hawaiian_to_english = self.convert_dict_no_diacritics(newdic)

    def remove_diacritics(self, text):
        """
        Remove diacritic characters from a string using the replacements map.
        """
        return "".join(self.replacements.get(c, c) for c in text)

    def convert_dict_no_diacritics(self, dic):
        """
        Converts dictionary keys into a format where diacritics are removed,
        allowing for flexible word lookups.
        """
        newdic = {}
        for key, translation_str in dic.items():
            no_diacritics = self.remove_diacritics(key)
            if no_diacritics not in newdic:
                newdic[no_diacritics] = []
            newdic[no_diacritics].append((key, translation_str))
        return newdic

    def load_pages(self, bookID):
        """
        Load the full list of pages (metadata) for a book ID.
        Cached after first access.
        """
        pgs = self.pages.get(bookID)
        if not pgs:
            with open(f"/data2/kahupuke/books/{bookID}/pages.json") as fin:
                pgs = json.load(fin)
            self.pages[bookID] = pgs
        return pgs

    def load_page_text(self, bookID, pageID):
        """
        Loads the full text of a single page, normalizes ʻokina,
        strips HTML tags, and caches the result.
        """
        try:
            cache_key = f"{bookID}.{pageID}"
            if not self.loaded_pages.get(cache_key):
                with open(f"/data2/kahupuke/books/{bookID}/text/{pageID}.txt") as fin:
                    orig = fin.read()
                orig = normalize_okina(orig)

                doc = BeautifulSoup("<xml>" + orig + "</xml>", "html.parser")

                text = ""
                for child in doc:
                    for subchild in child:
                        text += subchild.get_text().strip() + " "

                self.loaded_pages[cache_key] = doc.get_text()

            return self.loaded_pages.get(cache_key)
        except Exception as e:
            print(e)
            return str(e)


    def get_book_list(self):
        """
        Returns a formatted list of available books.
        """
        return self.books


    def get_book_cover_paths(self) -> dict[str, str]:
        """
        Build a {book_id: cover_path} mapping for every book in ``self.books``.
        We assume the directory listing order reflects the intended cover order,
        so we simply take the **first** image file returned by ``os.listdir``.
        """
        cover_lookup: dict[str, str] = {}

        for book in self.books:
            book_id  = book["id"]
            book_dir = f"/data2/kahupuke/images/{book_id}"

            cover_lookup[book_id] = ""

            try:
                for fn in os.listdir(book_dir):
                    if fn.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
                        cover_lookup[book_id] = os.path.join(book_dir, fn)
                        break
            except FileNotFoundError:
                pass

        return cover_lookup
