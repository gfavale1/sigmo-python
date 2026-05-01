import sigmo.matcher as sm

res = sm.match("C=O", "CC=O", iterations = 0)

print(f"Risultato: {res}")