# 全探索
# 選ぶ数が決まっている時に使用する

n = 5
for i in range(n):
    for j in range(i + 1, n):
        for k in range(j + 1, n):
            print(i, j, k)

"""
0 1 2
0 1 3
0 1 4
0 2 3
0 2 4
0 3 4
1 2 3
1 2 4
1 3 4
2 3 4
"""

# 左シフト
# 指定した桁だけ左にずらして、空いた桁には0を入れる

print(1 << 4)  # => 16

# 論理積
# 両方が1の時に1を入れる

print(5 & 4)  # => 4

# bit全探索
# 選ぶ数が決まっていない時に使用する => 全ての選択肢を選んでもいいし、選ばなくてもいい時に使用する

n = 4
for i in range(1 << n):
    use = [False] * n
    for j in range(n):
        if i & (1 << j):
            use[j] = True
    print(use)

"""
[False, False, False, False]
[True, False, False, False]
[False, True, False, False]
[True, True, False, False]
[False, False, True, False]
[True, False, True, False]
[False, True, True, False]
[True, True, True, False]
[False, False, False, True]
[True, False, False, True]
[False, True, False, True]
[True, True, False, True]
[False, False, True, True]
[True, False, True, True]
[False, True, True, True]
[True, True, True, True]
"""

from itertools import product

list(product([0, 1], repeat=n))

"""
[(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 0, 1, 1), (0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 1, 0), (0, 1, 1, 1), (1, 0, 0, 0), (1, 0, 0, 1), (1, 0, 1, 0), (1, 0, 1, 1), (1, 1, 0, 0), (1, 1, 0, 1), (1, 1, 1, 0), (1, 1, 1, 1)]
"""

# 二分探索

from typing import List, Optional

def binary_search(numbers: List[int], value: int) -> Optional[int]:
    left, right = 0, len(numbers) - 1
    while left <= right:
        mid = (left + right) // 2
        if numbers[mid] == value:
            return mid
        elif numbers[mid] < value:
            left = mid + 1
        else:
            right = mid - 1
    return None

numbers = [0, 1, 5, 7, 9, 11, 15, 20, 24]
print(binary_search(numbers, 15))  # => 6
print(binary_search(numbers, 2))  # => None

# 累積和
# 最初からi番目までの総和

from itertools import accumulate

N = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
cumulative_sum_list = [i for i in accumulate(N)]
print(cumulative_sum_list)  # => [1, 3, 6, 10, 15, 21, 28, 36, 45, 55]

# 組み合わせ (重複なし)

from itertools import combinations

combs = list(combinations([1, 2, 3, 4, 5], 2))
print(
    combs
)  # => [(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)]

# 組み合わせ (重複あり)

from itertools import combinations_with_replacement

combs = list(combinations_with_replacement([1, 2, 3, 4, 5], 2))
print(
    combs
)  # => [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (2, 2), (2, 3), (2, 4), (2, 5), (3, 3), (3, 4), (3, 5), (4, 4), (4, 5), (5, 5)]

# 四捨五入
# Python3系ではround関数は「偶数への丸め」を行う関数へ変更されているので、標準ライブラリのdecimalモジュールのquantize関数を使用する

from decimal import Decimal, ROUND_HALF_UP

print(int(Decimal(5).quantize(Decimal("1E1"), rounding=ROUND_HALF_UP)))  # => 10

# 2組の総数
# 制約
# ❌ [(1, 1)]

print(5 * (5 - 1) // 2)  # => 10

# 配列からindexと要素のタプルを生成

print(list(enumerate([50, 60, 70])))  # => [(0, 50), (1, 60), (2, 70)]

# 二次元平面上で原点を中心に反時計回りに回転させた座標

a, b, d = map(int, input().split())

import math

d_rad = math.radians(d)

a_rotated = a * math.cos(d_rad) - b * math.sin(d_rad)
b_rotated = a * math.sin(d_rad) + b * math.cos(d_rad)

print(a_rotated, b_rotated)

# ルート

import math

print(math.sqrt(2))  # => 1.4142135623730951
