import random
from itertools import combinations
from cards import ranks, suits, card_str, make_deck, evaluate_5, evaluate_7, evaluate_fast


def parse(s):
    return [ranks.index(t[0]) * 4 + suits.index(t[1]) for t in s.split()]


def test_tiebreakers():
    better = [
        ("KS KD AH 7C 3D", "KH KC QH 7D 3S"),
        ("6S 5D 4H 3C 2D", "5S 4D 3H 2C AD"),
        ("3S 3D 3H 2C 2D", "2S 2D 2H AC AD"),
        ("AS AD 3H 3C 2D", "KS KD QH QC AD"),
        ("QS QD 9H 9C 5D", "QH QC 9S 9D 4D"),
        ("AH KH 9H 7H 2H", "AS KS 9S 7S 3D"),
        ("2S 3S 4S 5S 6S", "AS AD AH AC KD"),
    ]
    for a, b in better:
        assert evaluate_5(parse(a)) > evaluate_5(parse(b)), (a, b)
    assert evaluate_5(parse("AS KD QH JC TD")) == evaluate_5(parse("AH KS QC JD TS"))


def test_evaluator(trials=20000):
    deck = make_deck()
    category_counts = [0] * 9
    for _ in range(trials):
        seven = random.sample(deck, 7)
        subset_scores = []
        for hand in combinations(seven, 5):
            s = evaluate_5(hand)
            assert s is not None and 0 <= s >> 20 <= 8, [card_str(c) for c in hand]
            subset_scores.append(s)
        best = evaluate_7(seven)
        assert best == max(subset_scores), [card_str(c) for c in seven]
        shuffled = seven[:]
        random.shuffle(shuffled)
        assert evaluate_7(shuffled) == best, "order-dependent"
        assert evaluate_fast(seven) == best, ("fast evaluator disagrees", [card_str(c) for c in seven])
        assert evaluate_fast(seven[:5]) == evaluate_5(seven[:5])
        category_counts[best >> 20] += 1
    names = ["high card", "one pair", "two pair", "trips", "straight",
             "flush", "full house", "quads", "straight flush"]
    expected = [17.4, 43.8, 23.5, 4.83, 4.62, 3.03, 2.60, 0.168, 0.0311]
    print(f"{trials} hands, all invariants held (incl. evaluate_fast == evaluate_7).\n")
    print(f"{'category':<15}{'yours':>8}{'expected':>10}")
    for c in range(9):
        pct = 100 * category_counts[c] / trials
        print(f"{names[c]:<15}{pct:>7.2f}%{expected[c]:>9.2f}%")


if __name__ == "__main__":
    test_tiebreakers()
    test_evaluator()
