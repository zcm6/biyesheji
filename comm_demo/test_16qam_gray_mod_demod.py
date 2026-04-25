from __future__ import annotations

import math
import numpy as np
from pipeline import bits_to_ints, ints_to_bits, gray_to_binary, binary_to_gray

def hamming_distance(a, b):
    return sum(x != y for x, y in zip(a, b))

def main():
    order = 16
    width = int(round(math.log2(order)))

    side = int(round(math.sqrt(order)))
    dim_width = int(round(math.log2(side)))

    levels = np.arange(-(side - 1), side + 1, 2, dtype=np.float32)
    points = np.array(
        [complex(i, q) for q in levels[::-1] for i in levels],
        dtype=np.complex64
    )

    # 生成所有 4bit 输入：0000 ~ 1111
    original_bits = np.array(
        [[int(b) for b in format(i, f"0{width}b")] for i in range(order)],
        dtype=np.int32
    )

    # =========================
    # 调制端：16QAM Gray 映射
    # =========================
    bit_groups = original_bits.reshape(-1, width)

    q_bin = bits_to_ints(bit_groups[:, :dim_width].reshape(-1), dim_width)
    i_bin = bits_to_ints(bit_groups[:, dim_width:].reshape(-1), dim_width)

    q_gray = binary_to_gray(q_bin)
    i_gray = binary_to_gray(i_bin)

    indices = (q_gray * side + i_gray).astype(np.int32, copy=False)
    modulated_points = points[indices]

    print("========== 16QAM Gray 调制映射表 ==========")
    print("原始bits -> q_bin i_bin -> q_gray i_gray -> index -> 星座点")

    for bits, qb, ib, qg, ig, idx, pt in zip(
        bit_groups, q_bin, i_bin, q_gray, i_gray, indices, modulated_points
    ):
        bit_str = "".join(map(str, bits))
        print(
            f"{bit_str} -> "
            f"q_bin={qb}, i_bin={ib} -> "
            f"q_gray={qg}, i_gray={ig} -> "
            f"index={idx:2d} -> "
            f"point=({pt.real:.0f}, {pt.imag:.0f})"
        )

    # =========================
    # 解调端：16QAM 解 Gray 映射
    # 这里假设无噪声，detected_indices 就是调制得到的 indices
    # =========================
    detected_indices = indices.copy()

    q_gray_dec = detected_indices // side
    i_gray_dec = detected_indices % side

    q_bin_dec = gray_to_binary(q_gray_dec)
    i_bin_dec = gray_to_binary(i_gray_dec)

    q_bits = ints_to_bits(q_bin_dec, dim_width).reshape(-1, dim_width)
    i_bits = ints_to_bits(i_bin_dec, dim_width).reshape(-1, dim_width)

    decoded_bits = np.concatenate([q_bits, i_bits], axis=1).reshape(-1)
    decoded_bit_groups = decoded_bits.reshape(-1, width)

    print("\n========== 16QAM 解 Gray 映射验证 ==========")
    print("原始bits -> detected_index -> 解码bits -> 是否一致")

    demod_ok = True

    for src, idx, dec in zip(original_bits, detected_indices, decoded_bit_groups):
        src_str = "".join(map(str, src))
        dec_str = "".join(map(str, dec))
        ok = src_str == dec_str

        if not ok:
            demod_ok = False

        print(f"{src_str} -> index={idx:2d} -> {dec_str} -> {ok}")

    # =========================
    # 星座图相邻点 Gray 性质验证
    # =========================
    grid_bits = np.empty((side, side), dtype=object)

    for bits, idx in zip(original_bits, indices):
        q_index = idx // side
        i_index = idx % side
        grid_bits[q_index, i_index] = "".join(map(str, bits))

    print("\n========== 星座图位置上的 bits 分布 ==========")

    for row in grid_bits:
        print("   ".join(row))

    print("\n========== 水平/垂直相邻点汉明距离检查 ==========")

    neighbor_ok = True

    for q in range(side):
        for i in range(side):
            current_bits = grid_bits[q, i]

            if i + 1 < side:
                right_bits = grid_bits[q, i + 1]
                dist = hamming_distance(current_bits, right_bits)
                print(f"{current_bits} <-> {right_bits}，汉明距离 = {dist}")
                if dist != 1:
                    neighbor_ok = False

            if q + 1 < side:
                down_bits = grid_bits[q + 1, i]
                dist = hamming_distance(current_bits, down_bits)
                print(f"{current_bits} <-> {down_bits}，汉明距离 = {dist}")
                if dist != 1:
                    neighbor_ok = False

    print("\n========== 最终验证结果 ==========")
    print("调制-解调闭环验证：", "通过" if demod_ok else "不通过")
    print("相邻点 Gray 性质验证：", "通过" if neighbor_ok else "不通过")

    if demod_ok and neighbor_ok:
        print("结论：Codex 生成的 16QAM Gray 调制与解调逻辑正确。")
    else:
        print("结论：当前 16QAM Gray 映射/解映射存在问题。")


if __name__ == "__main__":
    main()