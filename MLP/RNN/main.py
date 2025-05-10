# đọc file fox_and_grapes.txt
with open("fox_and_grapes.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("Length of text:", len(text))
print("First 100 characters:\n", text[:100])