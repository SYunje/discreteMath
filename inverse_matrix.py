"""Matrix inversion program using determinant/adjugate and Gauss-Jordan methods."""
from __future__ import annotations

from copy import deepcopy
from typing import List, Tuple

Matrix = List[List[float]]


def read_int(prompt: str) -> int:
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("정수를 입력하세요.")


def read_matrix(n: int) -> Matrix:
    matrix: Matrix = []
    for row in range(n):
        while True:
            raw = input(f"행 {row + 1}의 {n}개 원소를 공백으로 구분하여 입력: ")
            try:
                values = [float(x) for x in raw.strip().split()]
            except ValueError:
                print("숫자를 입력하세요.")
                continue
            if len(values) != n:
                print(f"{n}개 값을 입력해야 합니다.")
                continue
            matrix.append(values)
            break
    return matrix


def copy_matrix(matrix: Matrix) -> Matrix:
    return [row[:] for row in matrix]


def minor(matrix: Matrix, i: int, j: int) -> Matrix:
    return [
        [elem for col, elem in enumerate(row) if col != j]
        for row_idx, row in enumerate(matrix)
        if row_idx != i
    ]


def determinant(matrix: Matrix) -> float:
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    det = 0.0
    for j, elem in enumerate(matrix[0]):
        sign = -1 if j % 2 else 1
        det += sign * elem * determinant(minor(matrix, 0, j))
    return det


def transpose(matrix: Matrix) -> Matrix:
    return [list(col) for col in zip(*matrix)]


def adjugate(matrix: Matrix) -> Matrix:
    n = len(matrix)
    cofactors: Matrix = []
    for i in range(n):
        cofactor_row = []
        for j in range(n):
            cofactor = ((-1) ** (i + j)) * determinant(minor(matrix, i, j))
            cofactor_row.append(cofactor)
        cofactors.append(cofactor_row)
    return transpose(cofactors)


def inverse_via_adjugate(matrix: Matrix) -> Tuple[bool, Matrix | None]:
    det = determinant(matrix)
    if abs(det) < 1e-12:
        return False, None
    adj = adjugate(matrix)
    inv = [[elem / det for elem in row] for row in adj]
    return True, inv


def gauss_jordan(matrix: Matrix) -> Tuple[bool, Matrix | None]:
    n = len(matrix)
    aug = [row + [1.0 if i == idx else 0.0 for idx in range(n)] for i, row in enumerate(copy_matrix(matrix))]

    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        pivot = aug[pivot_row][col]
        if abs(pivot) < 1e-12:
            return False, None
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        pivot = aug[col][col]
        aug[col] = [value / pivot for value in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if abs(factor) < 1e-12:
                continue
            aug[row] = [elem - factor * piv for elem, piv in zip(aug[row], aug[col])]

    inverse = [row[n:] for row in aug]
    return True, inverse


def pretty_print_matrix(matrix: Matrix) -> None:
    for row in matrix:
        print(" ".join(f"{value:10.6f}" for value in row))


def matrices_equal(a: Matrix, b: Matrix, tol: float = 1e-6) -> bool:
    for row_a, row_b in zip(a, b):
        for val_a, val_b in zip(row_a, row_b):
            if abs(val_a - val_b) > tol:
                return False
    return True


def main() -> None:
    print("역행렬 계산 프로그램")
    n = read_int("행렬의 크기 n을 입력하세요: ")
    if n <= 0:
        print("양의 정수를 입력해야 합니다.")
        return
    matrix = read_matrix(n)

    print("\n--- 행렬식/여인수법을 이용한 역행렬 ---")
    det_success, det_inverse = inverse_via_adjugate(matrix)
    if det_success and det_inverse is not None:
        pretty_print_matrix(det_inverse)
    else:
        print("행렬식이 0이어서 역행렬이 존재하지 않습니다.")

    print("\n--- 가우스-조던 소거법을 이용한 역행렬 ---")
    gj_success, gj_inverse = gauss_jordan(matrix)
    if gj_success and gj_inverse is not None:
        pretty_print_matrix(gj_inverse)
    else:
        print("가우스-조던 소거법으로 역행렬을 구할 수 없습니다 (특이 행렬).")

    if det_success and gj_success and det_inverse and gj_inverse:
        same = matrices_equal(det_inverse, gj_inverse)
        print("\n두 방법의 결과가 ", "동일합니다." if same else "다릅니다.")


if __name__ == "__main__":
    main()
