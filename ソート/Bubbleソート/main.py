# O(n ** 2)

from typing import List

def bubble_sort(numbers: List[int]) -> List[int]:
    numbers_length = len(numbers)
    for i in range(numbers_length):
        for j in range(numbers_length - 1 - i):
            if numbers[j] > numbers[j + 1]:
                numbers[j], numbers[j + 1] = numbers[j + 1], numbers[j]
    return numbers

if __name__ == "__main__":
    numbers = [2, 5, 1, 8, 7, 3]
    print(bubble_sort(numbers))

"""
[1, 2, 3, 5, 7, 8]
"""
