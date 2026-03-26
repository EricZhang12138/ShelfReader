"""
Fuzzy Text Scorer: Produces per-class similarity scores from OCR text
using fuzzy string matching against known class names.

Supports both English class names and Swedish product terms,
since the GroceryStoreDataset contains Swedish packaging.

No learnable parameters — this is a pure string-matching module.
"""

import csv
import numpy as np
from typing import Dict, List, Optional

from rapidfuzz import fuzz


# Swedish aliases for GroceryStoreDataset classes.
# Each class can have multiple aliases (Swedish product names, brand terms,
# common OCR fragments). The scorer matches against ALL aliases and takes
# the max score per class.
SWEDISH_ALIASES: Dict[str, List[str]] = {
    # ── Fruit ─────────────────────────────────────────────────────────
    "Golden-Delicious":     ["Golden Delicious"],
    "Granny-Smith":         ["Granny Smith"],
    "Pink-Lady":            ["Pink Lady"],
    "Red-Delicious":        ["Red Delicious"],
    "Royal-Gala":           ["Royal Gala"],
    "Avocado":              ["Avokado"],
    "Banana":               ["Banan"],
    "Kiwi":                 ["Kiwi", "Kiwigron"],
    "Lemon":                ["Citron"],
    "Lime":                 ["Lime"],
    "Mango":                ["Mango"],
    "Cantaloupe":           ["Cantaloupe", "Cantaloupemelon"],
    "Galia-Melon":          ["Galiamelon", "Galia Melon"],
    "Honeydew-Melon":       ["Honungsmelon", "Honeydew"],
    "Watermelon":           ["Vattenmelon"],
    "Nectarine":            ["Nektarin"],
    "Orange":               ["Apelsin"],
    "Papaya":               ["Papaya"],
    "Passion-Fruit":        ["Passionsfrukt", "Passion"],
    "Peach":                ["Persika"],
    "Anjou":                ["Anjou", "Päron Anjou"],
    "Conference":           ["Conference", "Päron Conference"],
    "Kaiser":               ["Kaiser", "Päron Kaiser"],
    "Pineapple":            ["Ananas"],
    "Plum":                 ["Plommon"],
    "Pomegranate":          ["Granatäpple"],
    "Red-Grapefruit":       ["Grapefrukt", "Röd Grapefrukt"],
    "Satsumas":             ["Satsumas", "Clementin", "Satsuma"],

    # ── Juice ─────────────────────────────────────────────────────────
    "Bravo-Apple-Juice":    ["Bravo Apple", "Bravo Äppeljuice", "Bravo Äpple"],
    "Bravo-Orange-Juice":   ["Bravo Apelsin", "Bravo Orange", "Bravo Apelsinjuice"],
    "God-Morgon-Apple-Juice":                   ["God Morgon Äpple", "God Morgon Apple"],
    "God-Morgon-Orange-Juice":                  ["God Morgon Apelsin", "God Morgon Orange"],
    "God-Morgon-Orange-Red-Grapefruit-Juice":   ["God Morgon Apelsin Grapefrukt",
                                                  "God Morgon Orange Grapefruit"],
    "God-Morgon-Red-Grapefruit-Juice":          ["God Morgon Grapefrukt",
                                                  "God Morgon Red Grapefruit"],
    "Tropicana-Apple-Juice":        ["Tropicana Apple", "Tropicana Äpple"],
    "Tropicana-Golden-Grapefruit":  ["Tropicana Grapefrukt", "Tropicana Golden Grapefruit"],
    "Tropicana-Juice-Smooth":       ["Tropicana Smooth"],
    "Tropicana-Mandarin-Morning":   ["Tropicana Mandarin"],

    # ── Dairy: Milk ───────────────────────────────────────────────────
    "Arla-Ecological-Medium-Fat-Milk":  ["Arla Eko Mellanmjölk", "Arla Eko Mjölk",
                                          "Arla Ekologisk Mellanmjölk"],
    "Arla-Lactose-Medium-Fat-Milk":     ["Arla Laktosfri Mellanmjölk", "Arla Laktosfri Mjölk",
                                          "Arla Laktosfri"],
    "Arla-Medium-Fat-Milk":             ["Arla Mellanmjölk", "Arla Mjölk"],
    "Arla-Standard-Milk":               ["Arla Standardmjölk", "Arla Mjölk Standardmjölk"],
    "Garant-Ecological-Medium-Fat-Milk":["Garant Eko Mellanmjölk", "Garant Ekologisk Mjölk"],
    "Garant-Ecological-Standard-Milk":  ["Garant Eko Standardmjölk", "Garant Ekologisk Standardmjölk"],

    # ── Dairy: Oat ────────────────────────────────────────────────────
    "Oatly-Natural-Oatghurt":   ["Oatly Havregurt", "Oatly Natural"],
    "Oatly-Oat-Milk":           ["Oatly Havredryck", "Oatly Oat"],

    # ── Dairy: Sour ───────────────────────────────────────────────────
    "Arla-Ecological-Sour-Cream":   ["Arla Eko Gräddfil", "Arla Ekologisk Gräddfil",
                                      "Eko Gräddfil"],
    "Arla-Sour-Cream":              ["Arla Gräddfil", "Gräddfil"],
    "Arla-Sour-Milk":               ["Arla Filmjölk", "Filmjölk"],

    # ── Dairy: Soy ────────────────────────────────────────────────────
    "Alpro-Blueberry-Soyghurt":  ["Alpro Blåbär", "Alpro Blueberry"],
    "Alpro-Vanilla-Soyghurt":    ["Alpro Vanilj", "Alpro Vanilla"],
    "Alpro-Fresh-Soy-Milk":      ["Alpro Soja", "Alpro Soya", "Alpro Fresh"],
    "Alpro-Shelf-Soy-Milk":      ["Alpro Soja", "Alpro Soya", "Alpro Shelf"],

    # ── Dairy: Yoghurt ────────────────────────────────────────────────
    "Arla-Mild-Vanilla-Yoghurt":            ["Arla Mild Vanilj", "Arla Vanilj Yoghurt"],
    "Arla-Natural-Mild-Low-Fat-Yoghurt":    ["Arla Mild Naturell", "Arla Naturell Lättmjölk",
                                              "Arla Mild Naturell Yoghurt"],
    "Arla-Natural-Yoghurt":                 ["Arla Naturell Yoghurt", "Arla Naturell"],
    "Valio-Vanilla-Yoghurt":                ["Valio Vanilj", "Valio Vanilla"],
    "Yoggi-Strawberry-Yoghurt":             ["Yoggi Jordgubb", "Yoggi Strawberry"],
    "Yoggi-Vanilla-Yoghurt":                ["Yoggi Vanilj", "Yoggi Vanilla"],

    # ── Vegetables ────────────────────────────────────────────────────
    "Asparagus":            ["Sparris"],
    "Aubergine":            ["Aubergine"],
    "Cabbage":              ["Vitkål", "Kål"],
    "Carrots":              ["Morot", "Morötter"],
    "Cucumber":             ["Gurka"],
    "Garlic":               ["Vitlök"],
    "Ginger":               ["Ingefära"],
    "Leek":                 ["Purjolök", "Purjo"],
    "Brown-Cap-Mushroom":   ["Champinjon", "Svamp", "Portabello"],
    "Yellow-Onion":         ["Gul Lök", "Gullök", "Lök"],
    "Green-Bell-Pepper":    ["Grön Paprika", "Paprika Grön"],
    "Orange-Bell-Pepper":   ["Orange Paprika", "Paprika Orange"],
    "Red-Bell-Pepper":      ["Röd Paprika", "Paprika Röd"],
    "Yellow-Bell-Pepper":   ["Gul Paprika", "Paprika Gul"],
    "Floury-Potato":        ["Mjölig Potatis", "Potatis Mjölig"],
    "Solid-Potato":         ["Fast Potatis", "Potatis Fast"],
    "Sweet-Potato":         ["Sötpotatis"],
    "Red-Beet":             ["Rödbeta", "Rödbetor"],
    "Beef-Tomato":          ["Bifftomat"],
    "Regular-Tomato":       ["Tomat"],
    "Vine-Tomato":          ["Kvistomat", "Tomat Kvist"],
    "Zucchini":             ["Zucchini", "Squash"],
}


class FuzzyTextScorer:
    """
    Scores OCR text against each class name using fuzzy string matching.

    For each sample, produces a [num_classes] vector of similarity scores
    in [0, 1] that can be combined with image classifier logits.

    Matches against both English class names and Swedish aliases,
    taking the max score across all aliases for each class.

    Architecture (text path):
        OCR string → fuzzy match vs (English + Swedish) names → score vector [81]
    """

    def __init__(self, class_names: List[str], aliases: Optional[Dict[str, List[str]]] = None):
        self.class_names_raw = class_names
        self.num_classes = len(class_names)

        # Build list of all match targets per class:
        # [class_idx] → [normalized_name_1, normalized_name_2, ...]
        aliases = aliases or SWEDISH_ALIASES
        self.match_targets: List[List[str]] = []
        for name in class_names:
            targets = [name.replace("-", " ").lower()]
            if name in aliases:
                targets.extend(a.lower() for a in aliases[name])
            # Deduplicate while preserving order
            seen = set()
            unique = []
            for t in targets:
                if t not in seen:
                    seen.add(t)
                    unique.append(t)
            self.match_targets.append(unique)

        total_aliases = sum(len(t) for t in self.match_targets)
        print(f"[FuzzyTextScorer] {self.num_classes} classes, "
              f"{total_aliases} match targets (English + Swedish)")

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        aliases: Optional[Dict[str, List[str]]] = None,
    ) -> "FuzzyTextScorer":
        """Load class names from GroceryStoreDataset classes.csv."""
        class_names = []
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)  # skip header
            for row in reader:
                name = row[0].strip()
                class_id = int(row[1].strip())
                while len(class_names) <= class_id:
                    class_names.append("")
                class_names[class_id] = name
        return cls(class_names, aliases=aliases)

    def score(self, ocr_text: str) -> np.ndarray:
        """
        Compute fuzzy match scores between OCR text and all class names/aliases.

        For each class, matches against all aliases (English + Swedish)
        and takes the maximum score.

        Args:
            ocr_text: Raw OCR string from the image

        Returns:
            scores: [num_classes] array with values in [0, 1]
        """
        scores = np.zeros(self.num_classes, dtype=np.float32)

        text = ocr_text.strip().lower()
        if not text or text == "[unk]":
            return scores

        for i, targets in enumerate(self.match_targets):
            best = 0.0
            for target in targets:
                s = fuzz.token_set_ratio(text, target) / 100.0
                if s > best:
                    best = s
            scores[i] = best

        return scores

    def score_batch(self, ocr_texts: List[str]) -> np.ndarray:
        """Score a batch of OCR texts. Returns [B, num_classes]."""
        return np.array([self.score(t) for t in ocr_texts])
