"""Fixed test material: prompts, documents, tools, and the quality set. Nothing here is random at run time
except where a nonce is needed to defeat the prefix cache (and then the nonce is passed in by the caller)."""
import json
import random
import re

MODEL = "inkling-2-mini"

# The panel used by the benchmark the README's figures come from (bench.py of the inference-optimisation skill):
# speed depends on the text, so the same prompts are used here. The README's single-stream figure is the median
# over the first three.
PERF_PROMPTS = [
    "Write a Python implementation of an LRU cache with O(1) get and put, with docstrings and tests.",
    "Explain, step by step, how TCP congestion control works, including slow start and fast recovery.",
    "Write a short story (about 600 words) about a lighthouse keeper who finds a message in a bottle.",
    "Compare B-trees and LSM-trees for a write-heavy database workload. Use a table and give a recommendation.",
    "Write a bash script that backs up a directory to a timestamped tar.gz and keeps only the last 7 backups.",
    "Summarise the causes and consequences of the 1929 stock market crash in five paragraphs.",
]

LONG_OUTPUT_PROMPT = ("Write a very long and detailed essay about the history of bridge building, from antiquity to "
                      "the present day. Cover at least ten different eras. Do not stop early.")

DETERMINISM_PROMPTS = [
    "Explain in about 150 words how a hash table handles collisions.",
    "Write a Python function that returns the n-th Fibonacci number iteratively, then explain it briefly.",
    "List eight facts about the river Danube, one per line.",
    "Tell a short story about a fox and a crow.",
]

MULTI_TURN = [
    "I am planning a three-day trip to Lisbon. Suggest one activity for each day, one line per day.",
    "Replace the activity for day two with something indoors.",
    "Now give the whole plan again as a numbered list.",
    "Which of the three days involves the most walking? Answer in one sentence.",
    "Translate your last answer into French.",
]


def user(text):
    return [{"role": "user", "content": text}]


def req(messages, max_tokens=256, temperature=0, effort="none", **extra):
    p = {"model": MODEL, "messages": messages if isinstance(messages, list) else user(messages),
         "max_tokens": max_tokens, "temperature": temperature}
    if effort is not None:
        p["reasoning_effort"] = effort
    p.update(extra)
    return p


# --------------------------------------------------------------------------------------------- documents
_ADJ = "quiet amber distant narrow hollow bright crooked silver ancient restless gentle heavy pale wooden frozen".split()
_NOUN = ("harbour ledger orchard signal lantern glacier market bridge archive compass engine meadow furnace "
         "canal workshop").split()
_VERB = "records measures follows repairs crosses supplies watches divides carries shelters".split()
_PLACE = "the northern gate; the old mill; the tidal flats; the copper mine; the west tower; the salt road".split("; ")
COLOURS = ["crimson", "indigo", "saffron", "emerald", "ivory", "cobalt"]


def sentences(n, seed):
    """n filler sentences, deterministic for a seed. Each carries its own number so no two documents share text."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        out.append(f"Entry {seed}-{i}: the {rng.choice(_ADJ)} {rng.choice(_NOUN)} {rng.choice(_VERB)} the "
                   f"{rng.choice(_ADJ)} {rng.choice(_NOUN)} beside {rng.choice(_PLACE)}.")
    return out


def haystack(n_sentences, seed, needle_positions=(0.02, 0.5, 0.98)):
    """A document with one needle per position (fractions of its length). Returns (document, [(colour, code)])."""
    rng = random.Random(seed * 7919 + 1)
    sents = sentences(n_sentences, seed)
    needles = []
    for k, frac in enumerate(sorted(needle_positions, reverse=True)):
        colour = COLOURS[(len(needle_positions) - 1 - k) % len(COLOURS)]
        code = str(rng.randrange(100000, 999999))
        sents.insert(int(len(sents) * frac), f"Important: the access code for the {colour} vault is {code}.")
        needles.append((colour, code))
    needles.reverse()
    return f"[document {seed}]\n" + " ".join(sents), needles


def needle_question(doc, colour):
    return (f"{doc}\n\nQuestion about the document above: what is the access code for the {colour} vault? "
            f"Reply with the six-digit code only.")


# --------------------------------------------------------------------------------------------- unicode
UNICODE_CASES = [
    ("emoji", "🎉🚀 launch 👩‍👩‍👧‍👦 done 🇯🇵"),
    ("cjk", "黒潮は日本の太平洋岸を流れる暖流です。漢字とかなカナ。"),
    ("combining", "Zöé été ñ ǻ — café"),
    ("rtl", "שלום עולם مرحبا بالعالم"),
    ("mixed", "naïve Ærøskøbing Ω≈ç√∫ ½ ₿ ‰ „quoted“ — tab-free"),
    ("astral", "𝔘𝔫𝔦𝔠𝔬𝔡𝔢 𝓼𝓬𝓻𝓲𝓹𝓽 𐍈 𠜎"),
]


# --------------------------------------------------------------------------------------------- tools
def tool(name, description, properties=None, required=None, raw_parameters=...):
    f = {"name": name, "description": description}
    if raw_parameters is not ...:
        if raw_parameters is not None:
            f["parameters"] = raw_parameters
    else:
        f["parameters"] = {"type": "object", "properties": properties or {}, "required": required or []}
    return {"type": "function", "function": f}


S, N, I, B = {"type": "string"}, {"type": "number"}, {"type": "integer"}, {"type": "boolean"}
T = {
    "get_weather": tool("get_weather", "Get the current weather for a city.",
                        {"city": S, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, ["city"]),
    "calculator": tool("calculator", "Evaluate an arithmetic expression.", {"expression": S}, ["expression"]),
    "search_web": tool("search_web", "Search the web.", {"query": S, "max_results": I}, ["query"]),
    "send_email": tool("send_email", "Send an email.",
                       {"to": {"type": "array", "items": S}, "subject": S, "body": S}, ["to", "subject", "body"]),
    "create_event": tool("create_event", "Create a calendar event.", {
        "title": S,
        "start": {"type": "object", "properties": {"date": S, "time": S}, "required": ["date", "time"]},
        "attendees": {"type": "array", "items": {"type": "object", "properties": {"name": S, "email": S}}},
    }, ["title", "start"]),
    "get_time": tool("get_time", "Get the current local time. Takes no arguments."),
    "translate": tool("translate", "Translate text into a target language.", {"text": S, "target_lang": S}, ["text", "target_lang"]),
    "set_thermostat": tool("set_thermostat", "Set the thermostat.",
                           {"temperature": N, "mode": {"type": "string", "enum": ["heat", "cool", "auto"]}}, ["temperature"]),
    "read_file": tool("read_file", "Read a file from disk.", {"path": S}, ["path"]),
    "run_sql": tool("run_sql", "Run a SQL query against the database.", {"query": S}, ["query"]),
    "add_todos": tool("add_todos", "Add items to the todo list.", {
        "items": {"type": "array", "items": {"type": "object", "properties": {"text": S, "priority": I, "done": B},
                                             "required": ["text", "priority", "done"]}}}, ["items"]),
    "lookup_stock": tool("lookup_stock", "Get the latest price for a stock ticker.", {"ticker": S}, ["ticker"]),
    "post_message": tool("post_message", "Post a message to a chat channel.", {"channel": S, "text": S}, ["channel", "text"]),
    "list_users": tool("list_users", "List all users. Takes no arguments.", raw_parameters=None),
    "convert_currency": tool("convert_currency", "Convert an amount between currencies.",
                             {"amount": N, "from": S, "to": S}, ["amount", "from", "to"]),
    "bash": tool("bash", "Run a shell command and return its output.", {"command": S}, ["command"]),
    "edit_file": tool("edit_file", "Replace old_string with new_string in a file.",
                      {"path": S, "old_string": S, "new_string": S}, ["path", "old_string", "new_string"]),
    "plan_route": tool("plan_route", "Plan a route through waypoints given as [latitude, longitude] pairs.",
                       {"waypoints": {"type": "array", "items": {"type": "array", "items": N}}}, ["waypoints"]),
    "set_flags": tool("set_flags", "Set feature flags. Each flag is true or false.",
                      {"flags": {"type": "object", "additionalProperties": B}}, ["flags"]),
    "get-user_info2": tool("get-user_info2", "Get information about a user by id.", {"user_id": I}, ["user_id"]),
    "fetch_quarterly_financial_report_summary_for_company": tool(
        "fetch_quarterly_financial_report_summary_for_company", "Fetch a quarterly financial report summary.",
        {"company": S, "quarter": S, "year": I}, ["company", "quarter", "year"]),
    "save_note": tool("save_note", "Save a note.", {"text": S}, ["text"]),
    "create_order": tool("create_order", "Create an order.", {
        "customer": {"type": "object", "properties": {
            "name": S, "address": {"type": "object", "properties": {"street": S, "city": S, "zip": S}}}},
        "lines": {"type": "array", "items": {"type": "object", "properties": {"sku": S, "qty": I}}},
        "gift": B}, ["customer", "lines", "gift"]),
    "get_user_id": tool("get_user_id", "Look up the numeric id of a user by name.", {"name": S}, ["name"]),
    "get_orders": tool("get_orders", "List the orders of a user by numeric id.", {"user_id": I}, ["user_id"]),
}


def _has(path, pred):
    """Argument check: follow `path` (keys / indexes) and apply pred."""
    def f(args):
        cur = args
        try:
            for p in path:
                cur = cur[p]
        except (KeyError, IndexError, TypeError):
            return False
        try:
            return bool(pred(cur))
        except Exception:
            return False
    return f


def _eq(v):
    return lambda x: (str(x).strip().lower() == str(v).lower()) if isinstance(v, str) else x == v


def _contains(v):
    return lambda x: v.lower() in str(x).lower()


# (id, declared tools, messages, expected tool, argument checks). The argument checks are the "did it understand"
# part and are reported separately from well-formedness.
TOOL_CASES = [
    ("weather", ["get_weather"], "What's the weather in Paris, in celsius?", "get_weather",
     [_has(["city"], _contains("paris")), _has(["unit"], _eq("celsius"))]),
    ("calculator", ["calculator"], "Use the calculator to compute 1234 * 5678.", "calculator",
     [_has(["expression"], lambda x: "1234" in x and "5678" in x)]),
    ("search-int", ["search_web"], "Search the web for 'rust async runtime comparison' and give me at most 3 results.", "search_web",
     [_has(["query"], _contains("rust")), _has(["max_results"], _eq(3))]),
    ("email-array", ["send_email"], "Email ana@example.com and bob@example.com with the subject 'Standup moved' and the body 'Standup is at 10:30 tomorrow.'", "send_email",
     [_has(["to"], lambda x: isinstance(x, list) and set(x) == {"ana@example.com", "bob@example.com"}), _has(["subject"], _contains("standup moved"))]),
    ("event-nested", ["create_event"], "Create an event titled 'Design review' on 2026-12-01 at 14:00 with attendees Mia (mia@example.com) and Leo (leo@example.com).", "create_event",
     [_has(["start", "date"], _eq("2026-12-01")), _has(["attendees"], lambda x: isinstance(x, list) and len(x) == 2 and all(isinstance(a, dict) for a in x))]),
    ("empty-args", ["get_time"], "What time is it right now? Use the tool.", "get_time", [lambda a: a == {}]),
    ("unicode-cjk", ["translate"], "Use the translate tool to translate the Japanese text 'こんにちは世界' into English.", "translate",
     [_has(["text"], lambda x: "こんにちは世界" in x)]),
    ("number-enum", ["set_thermostat"], "Set the thermostat to 21.5 degrees in heat mode.", "set_thermostat",
     [_has(["temperature"], _eq(21.5)), _has(["mode"], _eq("heat"))]),
    ("path", ["read_file"], "Read the file /home/user/My Documents/notes (final).txt", "read_file",
     [_has(["path"], _eq("/home/user/My Documents/notes (final).txt"))]),
    ("sql-quotes", ["run_sql"], "Run this SQL exactly: SELECT name FROM users WHERE city = 'Zürich' AND note LIKE \"%x%\"", "run_sql",
     [_has(["query"], lambda x: "'Zürich'" in x and '"%x%"' in x)]),
    ("array-of-objects", ["add_todos"], "Add two todos: 'buy milk' with priority 2, not done; and 'file taxes' with priority 1, already done.", "add_todos",
     [_has(["items"], lambda x: len(x) == 2 and {i["text"].lower() for i in x} == {"buy milk", "file taxes"}),
      _has(["items"], lambda x: {(i["text"].lower(), i["priority"], i["done"]) for i in x} == {("buy milk", 2, False), ("file taxes", 1, True)})]),
    ("pick-of-3", ["get_weather", "lookup_stock", "calculator"], "What is the latest price of the ticker NVDA?", "lookup_stock",
     [_has(["ticker"], _eq("NVDA"))]),
    ("emoji", ["post_message"], "Post the message '🎉 Launch day! 🚀' to the #general channel.", "post_message",
     [_has(["text"], lambda x: "🎉" in x and "🚀" in x), _has(["channel"], _contains("general"))]),
    ("no-parameters-key", ["list_users"], "List all the users with the tool.", "list_users", []),
    ("currency", ["convert_currency", "calculator"], "Convert 250 US dollars to euros with the currency tool.", "convert_currency",
     [_has(["amount"], _eq(250)), _has(["from"], _contains("usd")), _has(["to"], _contains("eur"))]),
    ("bash", ["bash", "read_file"], "Use bash to list the files in /tmp sorted by size.", "bash",
     [_has(["command"], lambda x: "ls" in x and "/tmp" in x)]),
    ("multiline", ["edit_file"], "In the file app.py replace the text\n    return a - b\nwith\n    return a + b\nusing the edit tool.", "edit_file",
     [_has(["path"], _eq("app.py")), _has(["old_string"], _contains("return a - b")), _has(["new_string"], _contains("return a + b"))]),
    ("nested-arrays", ["plan_route"], "Plan a route through these waypoints: (48.85, 2.35), (50.11, 8.68) and (52.52, 13.40).", "plan_route",
     [_has(["waypoints"], lambda x: [[round(float(c), 2) for c in p] for p in x] == [[48.85, 2.35], [50.11, 8.68], [52.52, 13.4]])]),
    ("bool-map", ["set_flags"], "Turn the flag dark_mode on and the flag beta_search off.", "set_flags",
     [_has(["flags"], lambda x: x.get("dark_mode") is True and x.get("beta_search") is False)]),
    ("odd-name", ["get-user_info2", "get_weather"], "Get the information for the user with id 4471.", "get-user_info2",
     [_has(["user_id"], _eq(4471))]),
    ("long-name", ["fetch_quarterly_financial_report_summary_for_company", "search_web"], "Fetch the Q3 2025 financial report summary for Acme Corp.", "fetch_quarterly_financial_report_summary_for_company",
     [_has(["company"], _contains("acme")), _has(["year"], _eq(2025))]),
    ("rtl", ["save_note"], "Save a note with exactly this text: שלום עולם مرحبا", "save_note",
     [_has(["text"], lambda x: "שלום" in x and "مرحبا" in x)]),
    ("deep-nesting", ["create_order"], "Create a gift order for Ada Lovelace, 12 Byron Street, London, zip NW1 4AB, with 2 of SKU A-100 and 1 of SKU B-7.", "create_order",
     [_has(["customer", "address", "city"], _eq("London")), _has(["gift"], _eq(True)),
      _has(["lines"], lambda x: {(l["sku"], l["qty"]) for l in x} == {("A-100", 2), ("B-7", 1)})]),
    ("with-system", ["get_weather", "get_time"],
     [{"role": "system", "content": "You are a terse assistant. Use tools whenever they can answer the question."},
      {"role": "user", "content": "Is it raining in Tokyo?"}], "get_weather", [_has(["city"], _contains("tokyo"))]),
    ("after-history", ["calculator", "get_weather"],
     [{"role": "user", "content": "Hi! I will ask you for a few things."},
      {"role": "assistant", "content": "Sure, go ahead."},
      {"role": "user", "content": "Use the calculator for 99 * 99."}], "calculator", [_has(["expression"], lambda x: x.count("99") == 2)]),
    ("two-calls", ["get_weather"], "Get the weather in Paris and the weather in Tokyo (two tool calls).", "get_weather", []),
]


def tool_case_request(case, stream=False):
    cid, names, msgs, _exp, _checks = case
    return req(msgs if isinstance(msgs, list) else user(msgs), max_tokens=600, tools=[T[n] for n in names], stream=stream)


# --------------------------------------------------------------------------------------------- quality set
# Scoring rules (also in README.md):
#   - the answer is the `content` of the reply (thinking is ignored); it is normalised by trimming whitespace,
#     removing one trailing full stop, one surrounding pair of quotes / backticks / markdown bold markers (and a
#     full stop inside it), collapsing runs of whitespace and lower-casing;
#   - "exact": the normalised answer must equal the expected string (or one of the alternatives);
#   - "number": the normalised answer, with thousands separators removed, must be that number and nothing else
#     (a trailing unit is not accepted: the prompt says "only the number");
#   - "json": after removing one optional ```json fence the answer must parse as JSON and equal the expected value
#     (key order and whitespace do not matter);
#   - "nows": like exact, with all whitespace removed first (for comma-separated lists).
# A reply that ran out of tokens (finish_reason "length") or returned an error scores 0.
QUALITY = [
    # arithmetic
    ("arith-mul", "What is 17 * 23? Answer with only the number.", "number", 391),
    ("arith-add", "What is 1234 + 5678? Answer with only the number.", "number", 6912),
    ("arith-sub", "What is 1000 - 387? Answer with only the number.", "number", 613),
    ("arith-div", "What is 144 divided by 12? Answer with only the number.", "number", 12),
    ("arith-pow", "What is 2 to the power of 10? Answer with only the number.", "number", 1024),
    ("arith-mod", "What is the remainder when 100 is divided by 7? Answer with only the number.", "number", 2),
    ("arith-pct", "What is 15% of 200? Answer with only the number.", "number", 30),
    ("arith-sum", "What is the sum of the integers from 1 to 100? Answer with only the number.", "number", 5050),
    ("arith-dec", "What is 3.5 + 2.25? Answer with only the number.", "number", 5.75),
    ("arith-expr", "What is (8 + 4) * 5 - 10? Answer with only the number.", "number", 50),
    ("arith-min", "How many minutes are there in 2.5 hours? Answer with only the number.", "number", 150),
    ("arith-sqrt", "What is the square root of 81? Answer with only the number.", "number", 9),
    # strings
    ("str-reverse", "Reverse the string \"gauntlet\". Answer with only the reversed string.", "exact", "teltnuag"),
    ("str-upper", "Convert \"kuroshio\" to upper case. Answer with only the result.", "exact_case", "KUROSHIO"),
    ("str-len", "How many letters are in the word \"inference\"? Answer with only the number.", "number", 9),
    ("str-count", "How many times does the letter \"s\" appear in \"mississippi\"? Answer with only the number.", "number", 4),
    ("str-initials", "Take the first letter of each word in \"black current off japan\" and join them. Answer with only the result.", "exact", "bcoj"),
    ("str-replace", "Replace every letter \"a\" with \"o\" in the word \"banana\". Answer with only the result.", "exact", "bonono"),
    ("str-third", "What is the third word of the sentence \"the quick brown fox jumps\"? Answer with only that word.", "exact", "brown"),
    ("str-sort", "Sort these words alphabetically and separate them with single spaces: pear apple mango. Answer with only the sorted words.", "exact", "apple mango pear"),
    ("str-palindrome", "Is \"racecar\" a palindrome? Answer with only yes or no.", "exact", "yes"),
    ("str-concat", "Concatenate \"ink\" and \"ling\". Answer with only the result.", "exact", "inkling"),
    ("str-words", "How many words are in the sentence \"to be or not to be\"? Answer with only the number.", "number", 6),
    ("str-lower", "Convert \"HELLO World\" to lower case. Answer with only the result.", "exact_case", "hello world"),
    # logic
    ("logic-order", "Alice is taller than Bob. Bob is taller than Carol. Who is the shortest? Answer with only the name.", "exact", "carol"),
    ("logic-syllogism", "All bloops are razzies and all razzies are lazzies. Are all bloops lazzies? Answer with only yes or no.", "exact", "yes"),
    ("logic-seq", "What is the next number in the sequence 2, 4, 8, 16? Answer with only the number.", "number", 32),
    ("logic-day", "If today is Monday, what day of the week will it be in 3 days? Answer with only the day name.", "exact", "thursday"),
    ("logic-apples", "I have 3 apples, eat one, then buy 4 more. How many apples do I have? Answer with only the number.", "number", 6),
    ("logic-even", "True or false: 7 is an even number. Answer with only true or false.", "exact", "false"),
    ("logic-age", "Tom is older than Ann. Ann is older than Sue. Is Sue older than Tom? Answer with only yes or no.", "exact", "no"),
    ("logic-odd-one", "Which one is not a fruit: apple, banana, carrot, cherry? Answer with only the word.", "exact", "carrot"),
    ("logic-letter", "What is the next letter in the sequence A, C, E, G? Answer with only the letter.", "exact", "i"),
    ("logic-sort", "Sort these numbers in ascending order, separated by commas: 9, 3, 7, 1. Answer with only the numbers.", "nows", "1,3,7,9"),
    # format adherence
    ("fmt-json-person", "Ada is 36 years old. Answer with only a JSON object with keys \"name\" (string) and \"age\" (number).", "json", {"name": "Ada", "age": 36}),
    ("fmt-json-math", "Answer with only a JSON object with keys \"sum\" and \"product\" for the numbers 3 and 4.", "json", {"sum": 7, "product": 12}),
    ("fmt-json-array", "Answer with only a JSON array containing the first five prime numbers in ascending order.", "json", [2, 3, 5, 7, 11]),
    ("fmt-json-nested", "Answer with only a JSON object with keys \"ok\" (the boolean true) and \"items\" (an array of the strings \"a\" and \"b\").", "json", {"ok": True, "items": ["a", "b"]}),
    ("fmt-date", "Write the date 5 March 2021 in ISO 8601 format (YYYY-MM-DD). Answer with only the date.", "exact", "2021-03-05"),
    ("fmt-one-word", "What is the opposite of \"hot\"? Answer with exactly one word.", "exact", "cold"),
    ("fmt-csv", "Write the integers from 1 to 5 as one comma-separated line without spaces. Answer with only that line.", "exact", "1,2,3,4,5"),
    ("fmt-hyphen", "Join the words red, green and blue with hyphens, in that order, in lower case. Answer with only the result.", "exact", "red-green-blue"),
    ("fmt-hex", "Write the decimal number 255 in hexadecimal, lower case, with the 0x prefix. Answer with only the result.", "exact", "0xff"),
    ("fmt-yes", "Is Paris in France? Answer with only yes or no.", "exact", "yes"),
    # facts
    ("fact-capital-fr", "What is the capital of France? Answer with only the city name.", "exact", "paris"),
    ("fact-gold", "What is the chemical symbol for gold? Answer with only the symbol.", "exact_case", "Au"),
    ("fact-continents", "How many continents are there on Earth by the most common convention? Answer with only the number.", "number", 7),
    ("fact-mercury", "Which planet is closest to the Sun? Answer with only the planet's name.", "exact", "mercury"),
    ("fact-author", "Who wrote \"Romeo and Juliet\"? Answer with only the author's last name.", "exact", "shakespeare"),
    ("fact-boil", "At what temperature in degrees Celsius does water boil at sea level? Answer with only the number.", "number", 100),
    ("fact-leap", "How many days are there in a leap year? Answer with only the number.", "number", 366),
    ("fact-jupiter", "Which is the largest planet in the Solar System? Answer with only the planet's name.", "exact", "jupiter"),
    ("fact-ww2", "In what year did the Second World War end? Answer with only the year.", "number", 1945),
    ("fact-capital-jp", "What is the capital of Japan? Answer with only the city name.", "exact", "tokyo"),
    ("fact-h2o", "What is the common name of the compound H2O? Answer with only one word.", "exact", "water"),
    ("fact-sides", "How many sides does a hexagon have? Answer with only the number.", "number", 6),
]


def normalise(ans, keep_case=False):
    def drop_stop(t):
        t = t.strip()
        return t[:-1].rstrip() if t.endswith(".") else t
    a = drop_stop(ans)
    for l, r in (("**", "**"), ("`", "`"), ('"', '"'), ("'", "'"), ("“", "”")):
        if len(a) >= len(l) + len(r) and a.startswith(l) and a.endswith(r):
            a = drop_stop(a[len(l):len(a) - len(r)])
            break
    a = re.sub(r"\s+", " ", a)
    return a if keep_case else a.lower()


def score_quality(kind, expected, answer):
    """True if `answer` is correct under the rules above."""
    if kind == "exact":
        alts = expected if isinstance(expected, (list, tuple)) else [expected]
        return normalise(answer) in [str(x).lower() for x in alts]
    if kind == "exact_case":
        return normalise(answer, keep_case=True) == expected
    if kind == "nows":
        return re.sub(r"\s+", "", normalise(answer)) == expected
    if kind == "number":
        a = normalise(answer).replace(",", "").replace(" ", "")
        try:
            return abs(float(a) - float(expected)) < 1e-9
        except ValueError:
            return False
    if kind == "json":
        a = answer.strip()
        m = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", a, re.S)
        if m:
            a = m.group(1)
        try:
            return json.loads(a) == expected
        except ValueError:
            return False
    raise ValueError(kind)
