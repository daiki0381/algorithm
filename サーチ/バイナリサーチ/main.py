from typing import List

def binary_search(numbers: List[int], value: int) -> int:
    left, right = 0, len(numbers) - 1
    while left <= right:
        mid = (left + right) // 2
        if numbers[mid] == value:
            return mid
        elif numbers[mid] < value:
            left = mid + 1
        else:
            right = mid - 1
    return -1

def reflexive_binary_search(numbers: List[int], value: int) -> int:
    def _reflexive_binary_search(
        numbers: List[int], value: int, left: int, right: int
    ) -> int:
        if left > right:
            return -1
        mid = (left + right) // 2
        if numbers[mid] == value:
            return mid
        elif numbers[mid] < value:
            return _reflexive_binary_search(numbers, value, mid + 1, right)
        else:
            return _reflexive_binary_search(numbers, value, left, mid - 1)
    return _reflexive_binary_search(numbers, value, 0, len(numbers) - 1)

if __name__ == "__main__":
    numbers = [0, 1, 5, 7, 9, 11, 15, 20, 24]
    print(binary_search(numbers, 15))
    print(binary_search(numbers, 2))
    print(reflexive_binary_search(numbers, 15))
    print(reflexive_binary_search(numbers, 2))

"""
6
- 1
6
- 1
"""
