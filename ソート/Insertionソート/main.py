# O(n ** 2)
# 安定ソート

from typing import List

def insertion_sort(numbers: List[int]) -> List[int]:
    numbers_length = len(numbers)
    for i in range(1, numbers_length):
        temp = numbers[i]
        j = i - 1
        while j >= 0 and numbers[j] > temp:
            numbers[j + 1] = numbers[j]
            j -= 1
        numbers[j + 1] = temp
    return numbers

if __name__ == "__main__":
    numbers = [2, 5, 1, 8, 7, 3]
    print(insertion_sort(numbers))

"""
[1, 2, 3, 5, 7, 8]
"""
