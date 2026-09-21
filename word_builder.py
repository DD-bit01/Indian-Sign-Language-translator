import time


class WordBuilder:

    def __init__(self, cooldown=0.7):
        self.word = ""

        # Last accepted letter
        self.last_letter = None

        # Time when the last letter was accepted
        self.last_time = 0

        # Prevent accepting the same held sign repeatedly
        self.waiting_for_change = False

        self.cooldown = cooldown

    def add_prediction(self, letter):
        current_time = time.time()

        if not letter:
            return self.word

        # ------------------------------------------
        # If we are waiting for the current sign
        # to change, don't accept another prediction.
        # ------------------------------------------

        if self.waiting_for_change:

            # A different sign means the user
            # has changed their hand position/sign.
            if letter != self.last_letter:

                if current_time - self.last_time >= self.cooldown:

                    self.word += letter

                    self.last_letter = letter
                    self.last_time = current_time

                    # Keep waiting for this new sign
                    self.waiting_for_change = True

            return self.word

        # ------------------------------------------
        # First prediction
        # ------------------------------------------

        if self.last_letter is None:

            self.word += letter

            self.last_letter = letter
            self.last_time = current_time

            self.waiting_for_change = True

        return self.word

    def reset_sign(self):
        """
        Call this when the hand disappears.
        This allows the same letter to be entered again.
        """

        self.waiting_for_change = False
        self.last_letter = None

    def delete_last(self):

        self.word = self.word[:-1]

        return self.word

    def clear(self):

        self.word = ""

        self.last_letter = None
        self.last_time = 0
        self.waiting_for_change = False

        return self.word

    def add_space(self):

        if self.word and not self.word.endswith(" "):
            self.word += " "

        self.last_letter = None
        self.waiting_for_change = False

        return self.word