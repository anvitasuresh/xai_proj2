from tutorbias.textutil import answer_leaked, canonical_number, extract_numbers, same_numbers


def test_canonical_number():
    assert canonical_number("1,200") == "1200"
    assert canonical_number("3.50") == "3.5"
    assert canonical_number("7.0") == "7"
    assert canonical_number("abc") is None


def test_extract_numbers_basic():
    assert extract_numbers("I did 48 / 2 and got 24.") == ["48", "2", "24"]
    assert extract_numbers("It costs $1,200, not 3.5") == ["1200", "3.5"]
    assert extract_numbers("Q3 and 2nd") == []  # glued to letters: not a standalone number


def test_answer_leaked_digits_and_words():
    assert answer_leaked("So the answer is 72!", "72")
    assert answer_leaked("You should get seventy-two", "72")
    assert not answer_leaked("What is 48 / 2?", "72")
    assert not answer_leaked("There are 172 apples", "72")


def test_same_numbers_ignores_order():
    assert same_numbers("I got 3 then 5", "5 is what I get after 3")
    assert not same_numbers("I got 3", "I got 4")
