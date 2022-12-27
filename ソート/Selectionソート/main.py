from typing import List

def selection_sort(numbers: List[int]) -> List[int]:
    numbers_length = len(numbers)
    for i in range(numbers_length):
        min_index = i
        for j in range(i + 1, numbers_length):
            if numbers[min_index] > numbers[j]:
                min_index = j
        numbers[i], numbers[min_index] = numbers[min_index], numbers[i]
    return numbers

if __name__ == "__main__":
    numbers = [2, 5, 1, 8, 7, 3]
    print(selection_sort(numbers))

"""
[1, 2, 3, 5, 7, 8]
"""
