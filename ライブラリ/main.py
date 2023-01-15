# 全探索

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

print(1 << 4)  # => 16

# 論理積

print(5 & 4)  # => 4

# bit全探索
# 全ての選択肢に対して「使うか」「使わないか」を全列挙

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

# 順列全探索
# 順列 => 順序をつける並び方
# N！通りのパターンを全列挙

import itertools

print(
    list(itertools.permutations([0, 1, 2]))
)  # => [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]

# 二分探索


def binary_search(numbers, value):
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
# 最初からN番目までの総和

from itertools import accumulate

cumulative_sum = [i for i in accumulate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])]
print(cumulative_sum)  # => [1, 3, 6, 10, 15, 21, 28, 36, 45, 55]

# 約数全列挙
# 約数 => ある整数を割れる整数
# 例) 30 => 1、2、3、5、6、10、15、30


def make_divisors(n):
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    divisors.sort()
    return divisors


print(make_divisors(30))  # => [1, 2, 3, 5, 6, 10, 15, 30]

# 最大公約数 (GCD) => 共通する約数(ある整数を割れる整数)の中で一番大きな数
# 例) 24、36 => 12

import math

print(math.gcd(144, 264, 84, 276))  # => 12

# 最小公倍数 (LCM) => 共通する倍数(ある整数を何倍かした数)の中で一番小さな数
# 例) 27、9、3 => 27

import math

print(math.lcm(27, 9, 3))  # => 27

# 素因数分解 => 自然数(0を含まない正の整数)を素数(自然数のうち、1とその数でしか割り切れないもの)の積になるよう分解すること
# 例) 30 => 2、3、5


def make_prime_factorization(n):
    prime_factorization = []
    while n % 2 == 0:
        prime_factorization.append(2)
        n //= 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            prime_factorization.append(f)
            n //= f
        else:
            f += 2
    if n != 1:
        prime_factorization.append(n)
    return prime_factorization


print(make_prime_factorization(30))  # => [2, 3, 5]


# n以下の素数全列挙
# 素数 => 自然数のうち、1とその数でしか割り切れないもの
# エラトステネスの篩


def primes(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if not is_prime[i]:
            continue
        for j in range(i * 2, n + 1, i):
            is_prime[j] = False
    return [i for i in range(n + 1) if is_prime[i]]


print(primes(50))  # => [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

# 素数判定

import math


def is_prime(n):
    sqrt_n = math.ceil(math.sqrt(n))
    for i in range(2, sqrt_n):
        if n % i == 0:
            return False
    return True


print(is_prime(53))  # => True

# フィボナッチ数列 (動的計画法)
# フィボナッチ数列 => 初項と第2項を1とし、第3項以降は前2項の和となる数列 (1, 1, 2, 3, 5, 8, 13, 21, 34)
# 動的計画法 => 途中の計算結果を表の形式で保持しておくことで計算量を節約するアルゴリズム


def fibonacci(n):
    fibonacci_list = [1] * (n + 1)
    # 第3項以降は前2項の和
    for i in range(2, n + 1):
        fibonacci_list[i] = fibonacci_list[i - 1] + fibonacci_list[i - 2]
    return fibonacci_list[n - 1]


print(fibonacci(10))  # => 55

# フィボナッチ数列 (再帰)


def fibonacci(n):
    if n == 1 or n == 2:
        return 1
    return fibonacci(n - 1) + fibonacci(n - 2)


print(fibonacci(10))  # => 55

"""
nが3の場合
①fibonacci(2) + fibonacci(1)
②1 + 1 # => 2
"""

# 部分和 (動的計画法)
# 問 ) それぞれ5円、3円、2円の駄菓子が売ってあります。お小遣いを10円持っている時、最大何円分の駄菓子が買えるか (ただし、同じ種類の駄菓子はひとつまでしか買えないとします。)


def max_sum(num_list, limit):
    num_list_len = len(num_list)
    dp_list = [[0 for _ in range(limit + 1)] for _ in range(num_list_len)]

    # 1番目の商品
    for i in range(limit + 1):
        if i >= num_list[0]:
            dp_list[0][i] = num_list[0]

    # 2番目以降の商品
    for i in range(1, num_list_len):
        for j in range(limit + 1):
            # i番目の商品を買わなかった場合
            no_choice = dp_list[i - 1][j]
            if j < num_list[i]:
                dp_list[i][j] = no_choice
            else:
                # i番目の商品を買った場合
                # 1つ上の段のi番目の商品の金額分戻った列の金額+i番目の商品の金額
                choice = dp_list[i - 1][j - num_list[i]] + num_list[i]
                dp_list[i][j] = max(choice, no_choice)

    return dp_list[num_list_len - 1][limit]


num_list = [5, 3, 2]
limit = 10
print(max_sum(num_list, limit))  # => 10

# ナップサック問題 (bit全探索)
# 問 ) https://atcoder.jp/contests/dp/tasks/dp_d
# 問題点 => 計算量がO(2 ** N)なので、Nが増えると計算量が増大する => 今回の問題の制約は1≤N≤100なので、2 ** 100=1.2676506e+30となりTLE

from itertools import product

N, W = map(int, input().split())
wv = [list(map(int, input().split())) for _ in range(N)]
sum_values = []
patterns = product([0, 1], repeat=N)

for pattern in patterns:
    weight = 0
    value = 0
    for i in range(N):
        if pattern[i] == 1:
            weight += wv[i][0]
            value += wv[i][1]
    if weight <= W:
        sum_values.append(value)

print(max(sum_values))

# ナップサック問題 (動的計画法)
# 問 ) https://atcoder.jp/contests/dp/tasks/dp_d
# 計算量は表の縦 * 横となる => 今回の問題の制約は1≤N≤100 and 1≤W≤10 ** 5なので、100 * 10 ** 5=10 ** 7となり間に合う

N, W = map(int, input().split())
wv = [list(map(int, input().split())) for _ in range(N)]
dp = [[0] * (W + 1) for _ in range(N + 1)]

# 1番目
for i in range(W + 1):
    if i >= wv[0][0]:
        dp[1][i] = wv[0][1]

# 2番目以降
for i in range(2, N + 1):
    for j in range(W + 1):
        no_choice = dp[i - 1][j]
        if j < wv[i - 1][0]:
            dp[i][j] = no_choice
        else:
            # 1つ上の段のi番目のw分戻った列のv+i番目のv
            choice = dp[i - 1][j - wv[i - 1][0]] + wv[i - 1][1]
            dp[i][j] = max(no_choice, choice)

print(dp[N][W])

# BFS (幅優先探索)
# 全部回りたい時に使用する
# https://atcoder.jp/contests/abc277/tasks/abc277_c

from collections import deque, defaultdict

N = int(input())
d = defaultdict(list)

# ①ある頂点から行ける頂点を格納したリストを生成
for _ in range(N):
    a, b = map(int, input().split())
    d[a].append(b)
    d[b].append(a)

# ②キューを用意
que = deque()
que.append(1)

# ③訪問チェックを用意
seen = set()
seen.add(1)

# ④ 訪問後にque,seenに格納
while que:
    now = que.popleft()

    for i in d[now]:
        if i not in seen:
            que.append(i)
            seen.add(i)

print(max(seen))

# 連結成分の個数 (BFS)

""" 標準入力
5 3
1 2
1 3
4 5
"""

from collections import deque, defaultdict

N, M = map(int, input().split())
d = defaultdict(list)
ans = 0

# ①ある頂点から行ける頂点を格納したリストを生成
for _ in range(M):
    a, b = map(int, input().split())
    d[a].append(b)
    d[b].append(a)

# ②キューを用意
que = deque()

# ③訪問チェックを用意
seen = set()

# ④同じ連結成分に含まれる頂点を全て「探索済み」にする
for i in range(1, N + 1):
    if i not in seen:
        ans += 1
        que.append(i)
        while que:
            now = que.popleft()
            for i in d[now]:
                if i not in seen:
                    que.append(i)
                    seen.add(i)

print(ans)  # => 2


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

N = 5
print(int(Decimal(N).quantize(Decimal("1E1"), rounding=ROUND_HALF_UP)))  # => 10

# 2組の総数
# 制約
# ❌ [(1, 1)]

print(5 * (5 - 1) // 2)  # => 10

# 配列からindexと要素のタプルを生成

print(list(enumerate([50, 60, 70])))  # => [(0, 50), (1, 60), (2, 70)]

# 2次元配列内の1次元配列のn番目の要素に注目してソート

print(
    sorted([[0, 3], [1, 1], [2, 4], [3, 15], [4, 9]], key=lambda x: x[1])
)  # => [[1, 1], [0, 3], [2, 4], [4, 9], [3, 15]]

# 二次元平面上の2点間の距離

import math

A = (1, 2)
B = (3, 4)

print(math.sqrt((B[0] - A[0]) ** 2 + (B[1] - A[1]) ** 2))  # => 2.8284271247461903

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

# 部分文字列

print("aa" in "aabbcc")  # => True

# 1の位

print(1234 % 10)  # => 4

# 進数の変換

print(int("1011", 2))  # => 11

# Nまでの総和

N = 10
print(N * (N + 1) // 2)  # => 55

# forループをN-1のindexから逆順

N = 3
for n in reversed(range(N)):
    print(n)

"""
2
1
0
"""

# 配列の先頭と末尾の要素を取得

from collections import deque

q = deque([0, 1, 2, 3, 4, 5])
while q:
    print(q.popleft(), end=" ")
    print(q.pop(), end=" ")
    print()

"""
0 5
1 4
2 3
"""

# ラスト1周の算出
# https://atcoder.jp/contests/abc281/tasks/abc281_c
# 入 ) 3 600
# 入 ) 180 240 120

N, T = map(int, input().split())
A = list(map(int, input().split()))
T %= sum(A)
print(T)  # => 60

# 転置

print(
    list(zip(*[["#", ".", "#"], [".", "#", "."], ["#", ".", "#"]]))
)  # => [('#', '.', '#'), ('.', '#', '.'), ('#', '.', '#')]
print(
    list(zip(*["#.#", ".#.", "#.#"]))
)  # => [('#', '.', '#'), ('.', '#', '.'), ('#', '.', '#')]

# 特定の要素が入っているか
# 計算量 ) リスト => O(N)
# 計算量 ) 集合 => O(1)
# https://qiita.com/kitadakyou/items/6f877edd263f097e78f4

print(5 in {1, 2, 3, 4, 5, 6, 7, 8, 9, 10})  # => True

# defaultdict
# 存在しないキーを指定した場合、設定した関数の初期値が返却される

from collections import defaultdict

d = defaultdict(int)
d["name"] = "taro"
d["age"] = 20
print(d["name"])  # => taro
print(d["age"])  # => 20
print(d["gender"])  # => 0

# 配列内の要素数をハッシュ化

from collections import Counter

print(Counter([2, 7, 1, 8, 2, 8]))  # => Counter({2: 2, 8: 2, 7: 1, 1: 1})

# 部分文字列の個数

from itertools import combinations

s = "abc"
s_len = len(s)
substring_set = set()

for i in range(1, s_len + 1):
    combs = combinations(s, i)
    for comb in combs:
        substring_set.add(comb)

print(len(substring_set))  # => 7

# BMI
# 問 ) あなたは現在、ダイエット支援アプリを作ろうとしています。ユーザーが、現在の体重 w [kg] と現在の身長 h [cm] と目標BMI bを入力すると、最小何kg体重を落とすことで目標BMIを達成できるかを表示させる機能を作りたいです。BMIの計算式は BMI = 10000 * 体重 [kg] / 身長 [cm] / 身長 [cm] です。しかし現在、全ての入出力機能が非負整数にしか対応していません。そのため、例えば0.3kg体重を落とせば目標BMIを達成できるとしても必要量より少し多いですが非負整数である1kgの体重を落とすように表示しなければなりません。既に現在の体重、現在の身長で目標BMIを達成している場合は、0を出力してください。
# 標準入力 ) 90 180 23

import sys
import math


def main(lines):
    w, h, b = map(int, lines[0].split())
    # 現在のBMI
    current_bmi = 10000 * w / h / h
    if current_bmi <= b:
        print(0)
    else:
        # 目標BMIを達成させるための体重を算出
        goal_weight = b / 10000 * h * h
        # 落とす必要のある体重
        lose_weight = w - goal_weight
        print(math.ceil(lose_weight))  # => 16


if __name__ == "__main__":
    lines = []
    for l in sys.stdin:
        lines.append(l.rstrip("\r\n"))
    main(lines)

# アルファベット => 数値


def convert_alpha_to_num(alpha):
    num = 0
    for index, item in enumerate(list(alpha)):
        num += pow(26, len(alpha) - index - 1) * (ord(item) - ord("A") + 1)
    return num


print(convert_alpha_to_num("ALL"))  # => 1000
