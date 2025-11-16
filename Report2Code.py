# Relation Matrix / Equivalence Relation Checker
# A = {1, 2, 3, 4, 5}

N = 5  # 집합 A 크기 (고정)


# ----------------------------------------
# 1. 입력 & 출력 관련 함수
# ----------------------------------------
def input_matrix(n: int = N):
    print(f"{n}x{n} 관계 행렬을 입력하세요. (각 행을 0/1 공백 구분으로 입력)")
    mat = []
    for i in range(n):
        while True:
            line = input(f"행 {i+1}: ").strip().split()
            if len(line) != n:
                print(f"※ {n}개의 값을 입력해야 합니다. 다시 입력하세요.")
                continue
            try:
                row = [int(x) for x in line]
            except ValueError:
                print("※ 0과 1만 입력하세요.")
                continue
            if any(x not in (0, 1) for x in row):
                print("※ 0과 1만 입력하세요.")
                continue
            mat.append(row)
            break
    return mat


def print_matrix(mat):
    for row in mat:
        print(" ".join(str(x) for x in row))


# ----------------------------------------
# 2. 관계 성질 판별 함수
# ----------------------------------------
def is_reflexive(mat):
    n = len(mat)
    for i in range(n):
        if mat[i][i] != 1:
            return False
    return True


def is_symmetric(mat):
    n = len(mat)
    for i in range(n):
        for j in range(n):
            if mat[i][j] != mat[j][i]:
                return False
    return True


def is_transitive(mat):
    n = len(mat)
    for i in range(n):
        for j in range(n):
            if mat[i][j] == 1:
                for k in range(n):
                    if mat[j][k] == 1 and mat[i][k] == 0:
                        return False
    return True


def check_properties(mat, title="관계"):
    print(f"\n[{title}의 성질 판별]")
    r = is_reflexive(mat)
    s = is_symmetric(mat)
    t = is_transitive(mat)

    print(f"반사성 (Reflexive):   {'O' if r else 'X'}")
    print(f"대칭성 (Symmetric):   {'O' if s else 'X'}")
    print(f"추이성 (Transitive):   {'O' if t else 'X'}")

    if r and s and t:
        print("⇒ 이 관계는 동치 관계입니다.")
        return True
    else:
        print("⇒ 이 관계는 동치 관계가 아닙니다.")
        return False


# ----------------------------------------
# 3. 동치류 계산 함수
# ----------------------------------------
def equivalence_classes(mat):
    """
    mat가 동치 관계라고 가정하고,
    A = {1, ..., n} 에서 각 원소 i에 대해 [i] = { j | i R j } 를 반환
    """
    n = len(mat)
    classes = {}
    for i in range(n):
        cls = [j + 1 for j in range(n) if mat[i][j] == 1]
        classes[i + 1] = cls
    return classes


def print_equivalence_classes(mat, title="동치류"):
    print(f"\n[{title}]")
    classes = equivalence_classes(mat)
    for a in range(1, len(mat) + 1):
        cls = classes[a]
        cls_str = ", ".join(str(x) for x in cls)
        print(f"[{a}] = {{{cls_str}}}")


# ----------------------------------------
# 4. 폐포(closure) 함수들
# ----------------------------------------
def reflexive_closure(mat):
    n = len(mat)
    new = [row[:] for row in mat]
    for i in range(n):
        new[i][i] = 1
    return new


def symmetric_closure(mat):
    n = len(mat)
    new = [row[:] for row in mat]
    for i in range(n):
        for j in range(n):
            if new[i][j] == 1 or new[j][i] == 1:
                new[i][j] = 1
                new[j][i] = 1
    return new


def transitive_closure(mat):
    n = len(mat)
    new = [row[:] for row in mat]
    # Floyd–Warshall 스타일로 추이 폐포
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if new[i][k] == 1 and new[k][j] == 1:
                    new[i][j] = 1
    return new


# (추가기능 예시) 동치 관계 폐포
def equivalence_closure(mat):
    """
    R → R의 반사·대칭·추이 폐포를 모두 적용한
    '가장 작은 동치 관계'를 반환 (추가기능용)
    """
    r = reflexive_closure(mat)
    rs = symmetric_closure(r)
    rst = transitive_closure(rs)
    return rst


# ----------------------------------------
# 5. 메인 실행부
# ----------------------------------------
def main():
    # 1. 입력
    relation = input_matrix(N)

    print("\n[입력한 관계 행렬]")
    print_matrix(relation)

    # 2. 원래 관계의 성질 판별
    is_eq = check_properties(relation, "원래 관계")

    # 3. 동치 관계라면 동치류 출력
    if is_eq:
        print_equivalence_classes(relation, "원래 관계의 동치류")

    # 4. 각 폐포별 변환 전/후 + 다시 판별
    print("\n\n===== 반사 폐포 (Reflexive Closure) =====")
    reflexive = reflexive_closure(relation)
    print("[반사 폐포 변환 전]")
    print_matrix(relation)
    print("[반사 폐포 변환 후]")
    print_matrix(reflexive)
    if check_properties(reflexive, "반사 폐포"):
        print_equivalence_classes(reflexive, "반사 폐포의 동치류")

    print("\n\n===== 대칭 폐포 (Symmetric Closure) =====")
    symmetric = symmetric_closure(relation)
    print("[대칭 폐포 변환 전]")
    print_matrix(relation)
    print("[대칭 폐포 변환 후]")
    print_matrix(symmetric)
    if check_properties(symmetric, "대칭 폐포"):
        print_equivalence_classes(symmetric, "대칭 폐포의 동치류")

    print("\n\n===== 추이 폐포 (Transitive Closure) =====")
    transitive = transitive_closure(relation)
    print("[추이 폐포 변환 전]")
    print_matrix(relation)
    print("[추이 폐포 변환 후]")
    print_matrix(transitive)
    if check_properties(transitive, "추이 폐포"):
        print_equivalence_classes(transitive, "추이 폐포의 동치류")

    # (추가기능) 동치 관계 폐포 예시
    print("\n\n===== (추가기능) 동치 관계 폐포 =====")
    eq_closure = equivalence_closure(relation)
    print("[동치 관계 폐포 결과]")
    print_matrix(eq_closure)
    if check_properties(eq_closure, "동치 관계 폐포"):
        print_equivalence_classes(eq_closure, "동치 관계 폐포의 동치류")


if __name__ == "__main__":
    main()
