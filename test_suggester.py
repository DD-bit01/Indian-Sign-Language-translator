from word_suggester import WordSuggester

suggester = WordSuggester()

print("hi ->", suggester.suggest("hi"))
print("dilnawaz ->", suggester.suggest("dilnawaz"))
print("dilnwaz ->", suggester.suggest("dilnwaz"))
print("helo ->", suggester.suggest("helo"))