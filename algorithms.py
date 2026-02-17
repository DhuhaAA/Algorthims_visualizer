from __future__ import annotations
from typing import Generator, List, Tuple

# ------------------------------------------------------------
# Linear Search
# Time: O(n), Space: O(1)
# ------------------------------------------------------------
def linear_search(arr: List[int], target: int) -> int:
    for i, x in enumerate(arr):
        if x == target:
            return i
    return -1


# ------------------------------------------------------------
# Bubble Sort
# Best: O(n) with early exit; Avg/Worst: O(n^2); Space: O(1)
# ------------------------------------------------------------
def bubble_sort(arr: List[int]) -> List[int]:
    a = arr[:]
    n = len(a)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
                swapped = True
        if not swapped:
            break
    return a


def bubble_sort_steps(arr: List[int]) -> Generator[Tuple[List[int], Tuple[int, int]], None, None]:
    """Yields (array_snapshot, (i, j)) after each swap for visualization."""
    a = arr[:]
    n = len(a)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
                swapped = True
                yield a[:], (j, j + 1)
        if not swapped:
            break
    yield a[:], (-1, -1)


# ------------------------------------------------------------
# Merge Sort
# Best/Avg/Worst: O(n log n); Space: O(n)
# ------------------------------------------------------------
def merge_sort(arr: List[int]) -> List[int]:
    a = arr[:]
    if len(a) <= 1:
        return a
    mid = len(a) // 2
    left = merge_sort(a[:mid])
    right = merge_sort(a[mid:])
    return _merge(left, right)


def _merge(left: List[int], right: List[int]) -> List[int]:
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


# ------------------------------------------------------------
# Quick Sort (in-place with last-element pivot)
# Best/Avg: O(n log n); Worst: O(n^2); Space: O(log n) avg
# ------------------------------------------------------------
import random
from typing import List

def quick_sort(arr: List[int]) -> List[int]:
    a = arr[:]
    _quick_sort(a, 0, len(a) - 1)
    return a

def _quick_sort(a: List[int], low: int, high: int) -> None:
    while low < high:
        p = _partition_random_pivot(a, low, high)

        if (p - low) < (high - p):
            _quick_sort(a, low, p - 1)
            low = p + 1
        else:
            _quick_sort(a, p + 1, high)
            high = p - 1

def _partition_random_pivot(a: List[int], low: int, high: int) -> int:
    pivot_index = random.randint(low, high)
    a[pivot_index], a[high] = a[high], a[pivot_index]

    pivot = a[high]
    i = low - 1
    for j in range(low, high):
        if a[j] <= pivot:
            i += 1
            a[i], a[j] = a[j], a[i]
    a[i + 1], a[high] = a[high], a[i + 1]
    return i + 1

# ------------------------------------------------------------
# Radix Sort (LSD) for non-negative ints
# Time: O(nk); Space: O(n + k)
# ------------------------------------------------------------
def radix_sort(arr: List[int]) -> List[int]:
    a = arr[:]
    if len(a) == 0:
        return a
    if any(x < 0 for x in a):
        raise ValueError("Radix sort here supports non-negative integers only.")

    exp = 1
    max_val = max(a)
    while max_val // exp > 0:
        a = _counting_sort_by_digit(a, exp)
        exp *= 10
    return a


def _counting_sort_by_digit(a: List[int], exp: int) -> List[int]:
    n = len(a)
    output = [0] * n
    count = [0] * 10

    for x in a:
        digit = (x // exp) % 10
        count[digit] += 1

    for i in range(1, 10):
        count[i] += count[i - 1]

    for i in range(n - 1, -1, -1):
        digit = (a[i] // exp) % 10
        output[count[digit] - 1] = a[i]
        count[digit] -= 1

    return output