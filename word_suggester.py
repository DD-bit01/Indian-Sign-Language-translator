import difflib


class WordSuggester:

    def __init__(self):

        # Basic offline vocabulary for our first version
        self.words = [
            "hello",
            "hi",
            "hey",
            "good",
            "morning",
            "afternoon",
            "evening",
            "night",
            "dilnawaz",
            "team",
            "project",
            "hackathon",
            "we",
            "registered",
            "IQOO",
            

            "thank",
            "thanks",
            "welcome",
            "please",
            "sorry",

            "yes",
            "no",
            "okay",
            "goodbye",
            "bye",

            "water",
            "food",
            "home",
            "school",
            "college",
            "teacher",
            "student",
            "friend",
            "family",

            "apple",
            "book",
            "computer",
            "phone",
            "table",
            "chair",

            "india",
            "delhi",
            "kerala",
            "mumbai",
            "chennai",

            "what",
            "where",
            "when",
            "why",
            "who",
            "how",

            "my",
            "your",
            "name",
            "is",
            "are",
            "you",
            "i",
            "am",

            "help",
            "want",
            "need",
            "come",
            "go",
            "eat",
            "drink",
            "learn",
            "study",
            "work"
        ]


    def suggest(self, word, max_suggestions=3):

        """
        Return possible corrections for a detected word.
        """

        if not word:

            return []


        word = word.lower().strip()


        # Exact dictionary match
        if word in self.words:

            return [word]


        # Find similar words
        suggestions = difflib.get_close_matches(

            word,

            self.words,

            n=max_suggestions,

            cutoff=0.55
        )


        return suggestions