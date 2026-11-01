import random
import sys
from engine import GameState, play_hand, random_policy


def new(stacks=(100, 100), button=0):
    g = GameState(list(stacks), button, random.Random(1))
    g.start_hand()
    return g


def test_bb_option():
    g = new()
    g.apply_action("call")
    assert not g.street_over() and g.street == 0 and g.to_act == 1, "BB must get the option"
    assert g.legal_actions() == ["check", "raise"]
    g.apply_action("check")
    assert g.street == 1 and len(g.board) == 3 and g.to_act == 1, "flop: non-button acts first"
    assert g.street_bet == [0, 0] and g.contributed == [2, 2]


def test_raise_reopens():
    g = new()
    g.apply_action("call")
    g.apply_action("raise", 6)
    assert g.street == 0 and g.to_act == 0 and set(g.legal_actions()) == {"fold", "call", "raise"}
    assert g.min_raise_to() == 10


def test_shove_must_be_answered():
    g = new()
    g.apply_action("raise", 100)
    assert not g.hand_over and g.street == 0 and g.to_act == 1
    assert g.legal_actions() == ["fold", "call"], "no raising into an all-in"
    g.apply_action("call")
    assert g.hand_over and len(g.board) == 5 and sum(g.stacks) == 200


def test_shove_fold():
    g = new()
    g.apply_action("raise", 100)
    g.apply_action("fold")
    assert g.hand_over and g.net == [2, -2]


def test_short_stack_call_returns_excess():
    g = new(stacks=(100, 30))
    g.apply_action("raise", 100)
    g.apply_action("call")
    assert g.hand_over and sum(g.stacks) == 130
    assert g.stacks[0] >= 70, "uncalled 70 must go back to SB whatever the showdown"


def test_illegal_raise_rejected():
    g = new()
    for bad in (1, 3, 101):
        try:
            g.apply_action("raise", bad)
        except AssertionError:
            continue
        raise AssertionError(f"raise-to {bad} was accepted")


def test_chip_conservation(hands=100_000, seed=0):
    rng = random.Random(seed)
    pol = lambda g: random_policy(g, rng)
    for h in range(hands):
        stacks = [rng.randint(1, 300), rng.randint(1, 300)]
        g = play_hand([pol, pol], stacks=stacks, button=h % 2, rng=rng)
        assert sum(g.stacks) == sum(stacks), f"hand {h}: chips not conserved {g}"
        assert min(g.stacks) >= 0 and sum(g.net) == 0, f"hand {h}: {g}"
        assert len(g.board) == 5 or any(g.folded), f"hand {h}: ended without showdown or fold"
    print(f"{hands} random hands: chips conserved, no negative stacks, every hand ended.")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and name != "test_chip_conservation":
            fn()
            print("ok", name)
    test_chip_conservation(int(sys.argv[1]) if len(sys.argv) > 1 else 100_000)
