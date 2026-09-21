from word_builder import WordBuilder
import time

builder = WordBuilder(cooldown=1.0)

print("Adding A...")
print(builder.add_prediction("A"))

print("Trying A again immediately...")
print(builder.add_prediction("A"))

print("Waiting 1 second...")
time.sleep(1.1)

print("Adding P...")
print(builder.add_prediction("P"))

print("Waiting 1 second...")
time.sleep(1.1)

print("Adding P...")
print(builder.add_prediction("P"))

print("Waiting 1 second...")
time.sleep(1.1)

print("Adding L...")
print(builder.add_prediction("L"))

print("Waiting 1 second...")
time.sleep(1.1)

print("Adding E...")
print(builder.add_prediction("E"))

print("\nFinal word:")
print(builder.word)

print("\nDeleting last letter...")
print(builder.delete_last())

print("\nAdding space...")
print(builder.add_space())

print("\nClearing...")
print(builder.clear())