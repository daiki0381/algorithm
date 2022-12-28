from typing import List

def linear_search(numbers: List[int], value: int) -> int:
    for i in range(len(numbers)):
        if numbers[i] == value:
            return i
    return -1

if __name__ == "__main__":
    numbers = [0, 1, 5, 7, 9, 11, 15, 20, 24]
    print(linear_search(numbers, 15))
    print(linear_search(numbers, 2))

"""
6
- 1
"""
