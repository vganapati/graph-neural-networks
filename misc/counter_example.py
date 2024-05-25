from collections import Counter

counter_0 = Counter("mississippi")
counter_1 = Counter({"a": 0, "b": 1, "c": 2})

counter_1.update("a")

print(counter_1.most_common(1)) # most common
print(counter_1.most_common()) # list ordered from most to least common

breakpoint()
