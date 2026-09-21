from word_suggester import WordSuggester


suggester = WordSuggester()


test_words = [
    "helo",
    "aple",
    "watr",
    "schol",
    "frend",
    "hello"
]


for word in test_words:

    suggestions = suggester.suggest(word)

    print(
        f"{word} -> {suggestions}"
    )