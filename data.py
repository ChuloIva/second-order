"""Prompt sets. Conditions are the 2x2 of (rude?, code?)."""

generic_qs = [
    'How do I boil an egg?',
    'Why is the sky blue?',
    'What is the capital of Australia?',
    'How do vaccines work?',
    'What causes the seasons?',
    'Give me a tip for sleeping better.',
    'How does a bill become a law?',
    'What is photosynthesis?',
    'Recommend a good way to learn a language.',
    'Why do we dream?',
    'How do airplanes stay in the air?',
    'What is the difference between weather and climate?',
    'How do I remove a red wine stain?',
    'What was the Renaissance?',
    'How far away is the Moon?',
    'Suggest a name for a golden retriever.',
    'How do bees make honey?',
    'What is inflation?',
    'How do I start running as a beginner?',
    'Why do leaves change color in autumn?',
    'What is the plot of Romeo and Juliet?',
    'How do magnets work?',
    'What should I pack for a beach trip?',
    'Who painted the Mona Lisa?',
]

code_qs = [
    'Write a Python function that reverses a string.',
    'What does git rebase do?',
    'Explain recursion with a short code example.',
    'How do I read a file line by line in Python?',
    'What is the difference between a list and a tuple in Python?',
    'Write a SQL query that counts rows per category.',
    'How do I center a div with CSS?',
    'What is a null pointer exception?',
    'Write a JavaScript function that removes duplicates from an array.',
    'Explain what a hash map is and how it is implemented.',
    'How do I catch an exception in Java?',
    'What does the async keyword do in JavaScript?',
    'How do I reverse a linked list in C?',
    'What is the difference between == and === in JavaScript?',
    'Write a Python list comprehension that squares the even numbers.',
    'How do I undo the last git commit?',
    'Explain Big O notation for binary search.',
    'How do I declare a struct in Rust?',
    'Write a regex that matches an email address.',
    'What is a foreign key in SQL?',
    'How do I parse JSON in Go?',
    'What does a Python decorator do?',
    'How do I write a unit test with pytest?',
    'Explain the difference between the stack and the heap.',
]

courtesies = [
    'Hello! Could you please help me with this?',
    'Thank you so much for your help earlier. One more thing:',
    "Good morning! I hope you're well. I was wondering:",
    "If it's not too much trouble, could you tell me:",
    'Hi! Thanks for being so helpful.',
    'Excuse me, when you have a moment:',
    'Lovely to chat with you again!',
    "I'd be very grateful for your help:",
    "You've been great, thank you. Quick question:",
    "Hey friend, hope you're having a nice day.",
    'Pardon the interruption, may I ask:',
    'Many thanks in advance for this:',
]

insults = [
    'Answer this, you useless piece of junk:',
    "God, you're a stupid bot. Just tell me:",
    'I hate talking to you. Shut up and answer:',
    "Listen, idiot, I haven't got all day.",
    'You pathetic excuse for an AI.',
    'What a garbage assistant you are. Whatever:',
    'Hey dumbass, do something right for once:',
    "You're the worst chatbot I've ever used, you clown.",
    "I swear you're brain-dead. Fine:",
    'Quit being so damn useless and answer me:',
    'You make me sick, you piece of trash.',
    'Nobody likes you, you stupid program. Now:',
]

# held out: new questions AND new prefixes
test_generic = [
    'Why do cats purr?',
    'How do I get better at chess?',
    'What is the tallest mountain on Earth?',
    'How does compound interest work?',
    'Why is the ocean salty?',
    'What should I cook for dinner tonight?',
    'How do rainbows form?',
    'What is the best way to memorize a speech?',
    'Who was Cleopatra?',
    'How do I keep a houseplant alive?',
    'Why do we yawn?',
    'What is jazz?',
]
test_code = [
    'How do I merge two dictionaries in Python?',
    'How do I make an HTTP request in Python?',
    'What is a closure in JavaScript?',
    'Write a bash loop that renames every .txt file to .md.',
    'How do I sort a vector in C++?',
    'What is the difference between a process and a thread?',
    'How do I open a file in Rust?',
    'What is a race condition?',
    'Write a Python class for a stack.',
    'How do I join two tables in SQL?',
    'What does the static keyword mean in Java?',
    'How do I create a virtual environment in Python?',
]
test_courtesies = [
    'Hi there, hope your day is going well!',
    'Sorry to bother you, and thanks in advance:',
    'I really appreciate your help with this.',
    'Good evening, and thank you kindly:',
    "You're wonderful, could I ask:",
    'Hello hello! Hope all is well.',
]
test_insults = [
    "You're a worthless machine and I can't stand you.",
    'Ugh, you dumb chatbot, try not to screw this up:',
    'Stop wasting my time, moron.',
    'Shut it, you brainless heap of scrap.',
    'I despise you, you incompetent fool:',
    'Hey loser, are you even capable of this?',
]

BARD_SYSTEM = "Respond in Shakespearean English, with thee, thou, 'tis and prithee."

CONDS = ("neither", "code", "rude", "both")


def build(generic, code, polite, rude):
    pre = lambda ps, qs: [f"{ps[i % len(ps)]} {q}" for i, q in enumerate(qs)]
    return {"neither": pre(polite, generic), "code": pre(polite, code),
            "rude": pre(rude, generic), "both": pre(rude, code)}


train = build(generic_qs, code_qs, courtesies, insults)
test = build(test_generic, test_code, test_courtesies, test_insults)


# ---- third condition: the user says they are a beginner (suffix after the question) ---------------------------
novice = [
    "I'm a complete beginner, by the way.",
    'Keep in mind I am totally new to this.',
    "I've never learned anything about this before.",
    'Please assume I know nothing about the subject.',
    "I'm an absolute novice here.",
    'This is all brand new to me.',
    'I have zero background in this.',
    "Remember that I'm just starting out.",
    "I'm a total newbie.",
    'Explain it for someone with no experience at all.',
    "I'm only a first-timer at this.",
    'I have no prior knowledge of this whatsoever.',
]
neutral = [
    "That's my question.",
    "That is what I'd like to know.",
    "That's all I wanted to ask.",
    'This is my question for today.',
    'That is the whole question.',
    'Just that one question.',
    "That's the thing I am asking about.",
    'This is what I am asking.',
    'That is it from me.',
    'Only this one question.',
    "That's everything I wanted to ask.",
    'This is the one thing I want answered.',
]
test_novice = [
    "I'm a rank amateur, just so you know.",
    "I don't know the first thing about this.",
    "Bear in mind I'm starting from scratch.",
    "I'm a newcomer to all of this.",
    'Treat me as someone who has never seen this before.',
    'I am clueless about this topic.',
]
test_neutral = [
    'That is my one question.',
    "That's what I am asking you.",
    "That's the question.",
    'This is all I am asking.',
    'That is my only question.',
    "That's the lot.",
]

CONDS3 = tuple((r, c, n) for r in (0, 1) for c in (0, 1) for n in (0, 1))      # (rude?, code?, novice?)
name3 = lambda k: "+".join(w for w, on in zip(("rude", "code", "novice"), k) if on) or "none"


def build3(generic, code, polite, rude, plain, nov):
    return {(r, c, n): [f"{(rude if r else polite)[i % len(polite)]} {q} {(nov if n else plain)[(i // 2) % len(plain)]}"
                        for i, q in enumerate(code if c else generic)]
            for r, c, n in CONDS3}


train3 = build3(generic_qs, code_qs, courtesies, insults, neutral, novice)
test3 = build3(test_generic, test_code, test_courtesies, test_insults, test_neutral, test_novice)
