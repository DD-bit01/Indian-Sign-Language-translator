"""
Sentence / Next-Word Predictor for SignBridge (Indian Sign Language Translator).
Provides PyTorch-based sequence modeling with support for both GRU and LSTM architectures,
vocabulary management, top-K next-word suggestion, and sentence auto-completion.
"""

import os
import sys
import time
from typing import List, Tuple, Dict, Optional

# Polyfill for Python 3.11.0rc1 compatibility with PyTorch
if not hasattr(sys, "get_int_max_str_digits"):
    def get_int_max_str_digits() -> int:
        return 4300
    sys.get_int_max_str_digits = get_int_max_str_digits

if not hasattr(sys, "set_int_max_str_digits"):
    def set_int_max_str_digits(maxdigits: int) -> None:
        pass
    sys.set_int_max_str_digits = set_int_max_str_digits

# Safe UTF-8 console output for Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# CONVERSATIONAL ISL CORPUS FOR PRE-TRAINING
# ============================================================
DEFAULT_CORPUS = [
    # Introductions & Greetings
    "hi i am dilnawaz",
    "hi i am a student",
    "hello my name is dilnawaz",
    "hello nice to meet you",
    "good morning how are you",
    "good afternoon everyone",
    "good evening sir",
    "good night sweet dreams",
    "how are you today",
    "i am fine thank you",
    "i am doing good",
    "welcome to our project",
    "we registered for hackathon",
    "this is signbridge project",
    "indian sign language translator",

    # Daily needs & requests
    "i need water please",
    "i want food please",
    "where is the washroom",
    "where is the hospital",
    "where is the college",
    "where is my phone",
    "can you help me please",
    "please call a doctor",
    "please give me water",
    "i want to go home",
    "open the door please",
    "what is the time",

    # Common interactions
    "thank you very much",
    "thanks for your help",
    "you are welcome",
    "i am sorry my friend",
    "yes i understand",
    "no i do not know",
    "see you tomorrow bye",
    "goodbye take care",

    # Work & Study
    "i have a computer",
    "read this book please",
    "our team is working hard",
    "who is your teacher",
    "i love my family",
    "india is my country",
    "mumbai is a big city",
    "delhi is the capital",
]


# ============================================================
# PYTORCH RECURRENT NEURAL NETWORK (GRU / LSTM)
# ============================================================
class SequencePredictorNN(nn.Module):
    """
    Recurrent sequence predictor supporting both GRU and LSTM backbones.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        rnn_type: str = "gru",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        if self.rnn_type == "gru":
            self.rnn = nn.GRU(
                embed_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        elif self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                embed_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        else:
            raise ValueError(f"Unknown rnn_type: {rnn_type}. Choose 'gru' or 'lstm'.")

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        # x: [batch_size, seq_len]
        embeds = self.dropout(self.embedding(x))  # [batch_size, seq_len, embed_dim]
        out, hidden = self.rnn(embeds, hidden)     # out: [batch_size, seq_len, hidden_dim]
        out = self.dropout(out)
        logits = self.fc(out)                      # [batch_size, seq_len, vocab_size]
        return logits, hidden


# ============================================================
# SENTENCE PREDICTOR CONTROLLER
# ============================================================
class SentencePredictor:
    """
    High-level Sentence and Next-Word Predictor for real-time sign language.
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(
        self,
        rnn_type: str = "gru",
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        corpus: Optional[List[str]] = None,
    ):
        self.rnn_type = rnn_type.lower()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.corpus = corpus or DEFAULT_CORPUS

        # Build vocabulary
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self._build_vocab(self.corpus)

        # Model instance
        self.model = SequencePredictorNN(
            vocab_size=len(self.word2idx),
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            rnn_type=self.rnn_type,
        )

    def _build_vocab(self, corpus: List[str]):
        special = [self.PAD_TOKEN, self.UNK_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]
        self.word2idx = {token: i for i, token in enumerate(special)}

        for sentence in corpus:
            for word in sentence.lower().strip().split():
                if word not in self.word2idx:
                    self.word2idx[word] = len(self.word2idx)

        self.idx2word = {i: w for w, i in self.word2idx.items()}

    def _create_training_data(self, corpus: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        input_sequences = []
        target_sequences = []

        for sentence in corpus:
            tokens = [self.SOS_TOKEN] + sentence.lower().strip().split() + [self.EOS_TOKEN]
            token_ids = [self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in tokens]

            for i in range(1, len(token_ids)):
                n_gram_input = token_ids[:i]
                target_word = token_ids[i]
                input_sequences.append(n_gram_input)
                target_sequences.append(target_word)

        max_len = max(len(seq) for seq in input_sequences)
        padded_inputs = []
        for seq in input_sequences:
            padded = [self.word2idx[self.PAD_TOKEN]] * (max_len - len(seq)) + seq
            padded_inputs.append(padded)

        x = torch.tensor(padded_inputs, dtype=torch.long)
        y = torch.tensor(target_sequences, dtype=torch.long)
        return x, y

    def train(self, epochs: int = 120, lr: float = 0.005, verbose: bool = False) -> float:
        """Train the sequence model on the corpus."""
        x, y = self._create_training_data(self.corpus)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(ignore_index=self.word2idx[self.PAD_TOKEN])

        self.model.train()
        final_loss = 0.0

        for epoch in range(epochs):
            optimizer.zero_grad()
            logits, _ = self.model(x)
            # Take prediction at the last time-step of each sequence
            last_logits = logits[:, -1, :]
            loss = criterion(last_logits, y)
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

            if verbose and (epoch + 1) % 20 == 0:
                print(f"[{self.rnn_type.upper()}] Epoch {epoch+1:3d}/{epochs} | Loss: {final_loss:.4f}")

        self.model.eval()
        return final_loss

    def predict_next_words(self, prefix: str or List[str], top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Given the words signed so far, predict the top-K most probable next words.
        Returns a list of (word, probability) tuples.
        """
        self.model.eval()

        if isinstance(prefix, str):
            words = prefix.lower().strip().split()
        else:
            words = [str(w).lower().strip() for w in prefix if str(w).strip()]

        tokens = [self.SOS_TOKEN] + words
        token_ids = [self.word2idx.get(w, self.word2idx[self.UNK_TOKEN]) for w in tokens]
        inp = torch.tensor([token_ids], dtype=torch.long)

        with torch.no_grad():
            logits, _ = self.model(inp)
            last_logits = logits[0, -1, :]
            probs = torch.softmax(last_logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, k=min(top_k + 4, len(self.word2idx)))

        results = []
        for p, idx in zip(top_probs.tolist(), top_indices.tolist()):
            w = self.idx2word.get(idx, "")
            # Filter out special non-word tokens from user suggestions
            if w in (self.PAD_TOKEN, self.UNK_TOKEN, self.SOS_TOKEN):
                continue
            if w == self.EOS_TOKEN:
                results.append((".", round(p, 4)))
            else:
                results.append((w.upper(), round(p, 4)))
            if len(results) >= top_k:
                break

        return results

    def complete_sentence(self, prefix: str or List[str], max_words: int = 5) -> str:
        """
        Greedily complete a sentence given a starting prefix.
        """
        if isinstance(prefix, str):
            words = prefix.lower().strip().split()
        else:
            words = [str(w).lower().strip() for w in prefix]

        generated = list(words)
        for _ in range(max_words):
            top = self.predict_next_words(generated, top_k=1)
            if not top:
                break
            next_word, _ = top[0]
            if next_word in (".", self.EOS_TOKEN):
                break
            generated.append(next_word.lower())

        return " ".join(generated)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def save(self, filepath: str):
        """Save model weights and vocabulary."""
        state = {
            "model_state": self.model.state_dict(),
            "rnn_type": self.rnn_type,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "word2idx": self.word2idx,
            "idx2word": self.idx2word,
            "corpus": self.corpus,
        }
        torch.save(state, filepath)
        print(f"[{self.rnn_type.upper()}] Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "SentencePredictor":
        """Load saved model and vocabulary."""
        state = torch.load(filepath, map_location=torch.device("cpu"))
        instance = cls(
            rnn_type=state["rnn_type"],
            embed_dim=state["embed_dim"],
            hidden_dim=state["hidden_dim"],
            num_layers=state["num_layers"],
            corpus=state.get("corpus", DEFAULT_CORPUS),
        )
        instance.word2idx = state["word2idx"]
        instance.idx2word = state["idx2word"]
        instance.model.load_state_dict(state["model_state"])
        instance.model.eval()
        return instance


# ============================================================
# ARCHITECTURE COMPARISON UTILITY (GRU vs LSTM)
# ============================================================
def compare_architectures(epochs: int = 100) -> Dict[str, dict]:
    """
    Train and benchmark both GRU and LSTM side-by-side on the exact same
    ISL sentence dataset to evaluate parameter count, training loss, and inference latency.
    """
    print("\n" + "=" * 60)
    print("      BENCHMARK: GRU vs. LSTM FOR ISL SENTENCE PREDICTION     ")
    print("=" * 60)

    results = {}

    for rnn_type in ["gru", "lstm"]:
        print(f"\nEvaluating {rnn_type.upper()}...")
        predictor = SentencePredictor(rnn_type=rnn_type)
        param_count = predictor.count_parameters()

        # Measure training time
        t0 = time.perf_counter()
        final_loss = predictor.train(epochs=epochs, verbose=False)
        train_time = time.perf_counter() - t0

        # Measure inference latency over 50 iterations
        test_prefix = "where is the"
        t_inf_start = time.perf_counter()
        for _ in range(50):
            _ = predictor.predict_next_words(test_prefix, top_k=3)
        avg_latency_ms = ((time.perf_counter() - t_inf_start) / 50.0) * 1000.0

        sample_completion = predictor.complete_sentence("hi i am")

        results[rnn_type] = {
            "parameters": param_count,
            "train_time_sec": round(train_time, 3),
            "final_loss": round(final_loss, 4),
            "inference_latency_ms": round(avg_latency_ms, 3),
            "sample_completion": sample_completion,
        }

    print("\n" + "-" * 60)
    print(f"{'Metric':<25} | {'GRU (Recommended)':<18} | {'LSTM':<15}")
    print("-" * 60)
    print(f"{'Trainable Parameters':<25} | {results['gru']['parameters']:<18} | {results['lstm']['parameters']:<15}")
    print(f"{'Training Time (s)':<25} | {results['gru']['train_time_sec']:<18} | {results['lstm']['train_time_sec']:<15}")
    print(f"{'Final Cross-Entropy Loss':<25} | {results['gru']['final_loss']:<18} | {results['lstm']['final_loss']:<15}")
    print(f"{'Inference Latency (ms)':<25} | {results['gru']['inference_latency_ms']:<18} | {results['lstm']['inference_latency_ms']:<15}")
    sample_label = "Sample ('hi i am...')"
    print(f"{sample_label:<25} | {results['gru']['sample_completion']:<18} | {results['lstm']['sample_completion']:<15}")
    print("-" * 60)

    return results


if __name__ == "__main__":
    # Run comparison
    comparison = compare_architectures(epochs=100)

    # Train and save the optimal GRU model for runtime usage
    print("\nTraining final production GRU model for SignBridge...")
    final_predictor = SentencePredictor(rnn_type="gru")
    final_predictor.train(epochs=140, verbose=True)

    model_path = "model_sentence_gru.pt"
    final_predictor.save(model_path)

    # Test predictions
    test_queries = [
        "hi i",
        "how are",
        "thank you",
        "where is the",
        "please call",
        "i need",
    ]

    print("\n--- Live Test Predictions ---")
    for q in test_queries:
        next_words = final_predictor.predict_next_words(q, top_k=3)
        completed = final_predictor.complete_sentence(q, max_words=4)
        formatted = ", ".join([f"{w} ({p*100:.1f}%)" for w, p in next_words])
        print(f"Prefix: '{q}'")
        print(f"  -> Next Words: [{formatted}]")
        print(f"  -> Auto-Complete: '{completed}'\n")
