# O(1)

def func1(numbers):
    return numbers[0]

print(func1([0, 1, 2, 3, 4, 5]))

"""
0
"""

# O(log(n))

def func2(n):
    if n <= 1:
        return
    else:
        print(n)
        func2(n / 2)

func2(10)

"""
10
5.0
2.5
1.25
"""

# O(n)

def func3(numbers):
    for number in numbers:
        print(number)

func3([0, 1, 2, 3, 4, 5])

"""
0
1
2
3
4
5
"""

# O(n * log(n))

def func4(n):
    for i in range(int(n)):
        print(i, end=" ")
    print()

    if n <= 1:
        return
    else:
        func4(n / 2)

func4(10)

"""
0 1 2 3 4 5 6 7 8 9
0 1 2 3 4
0 1
0
"""

# O(n ** 2)

def func5(numbers):
    for i in range(len(numbers)):
        for j in range(len(numbers)):
            print(numbers[i], numbers[j])
        print()

func5([0, 1, 2, 3, 4, 5])

"""
0 0
0 1
0 2
0 3
0 4
0 5

1 0
1 1
1 2
1 3
1 4
1 5

2 0
2 1
2 2
2 3
2 4
2 5

3 0
3 1
3 2
3 3
3 4
3 5

4 0
4 1
4 2
4 3
4 4
4 5

5 0
5 1
5 2
5 3
5 4
5 5
"""
