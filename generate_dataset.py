import pandas as pd
import random

# Sample spam & ham message pools
spam_texts = [
    "Congratulations! You've won a free ticket to Bahamas! Call now to claim.",
    "URGENT! You have won a 1 week FREE membership in our £100,000 prize jackpot!",
    "Win a brand new iPhone! Reply WIN or visit our site to claim your reward.",
    "Claim your free ringtone by texting WIN to 80082! Offer ends soon.",
    "Get cheap loans now! Low interest, quick approval. Reply YES to apply.",
    "You’ve been selected to receive a $1000 Walmart gift card. Visit our link now.",
    "SIX chances to win CASH! From 100 to 20,000 pounds. Text WIN to 87575.",
    "WINNER!! As a valued customer you have been selected for a £900 reward!",
    "Your number has won 10,000 GBP in our lucky draw. Contact us immediately!",
    "You have won a lottery of $10,000! Send your details immediately to claim."
]

ham_texts = [
    "Hey, are we meeting later today?",
    "Can you send me the report by evening?",
    "Let's have lunch tomorrow at 1 pm.",
    "Call me when you are free.",
    "I'll reach office by 10am.",
    "Don't forget to buy groceries on your way home.",
    "Meeting got postponed to next week.",
    "Please find the attached document for your review.",
    "See you at the gym later!",
    "Did you finish the project presentation?"
]

# Generate large dataset
records = []
for _ in range(6000):  # change number here for more or fewer rows
    if random.random() > 0.3:  # 70% ham, 30% spam
        label = "ham"
        text = random.choice(ham_texts)
    else:
        label = "spam"
        text = random.choice(spam_texts)
    # Add random noise to vary text
    text += " " + random.choice(["ok", "pls", "call", "reply", "now", "urgent", "!!!", ""])
    records.append((label, text))

# Create and save
df = pd.DataFrame(records, columns=["Label", "EmailText"])
df.to_csv("spam.csv", index=False)
print("✅ Generated spam.csv with", len(df), "rows")
