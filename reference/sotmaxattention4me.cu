__global__ void flash_attn_kernel(
    const float *__restrict__ Q,
    const float *__restrict__ K,
    const float *__restrict__ V,
    float *__restrict__ O,
    int M, int N, int d)
{
    // ──── [THREAD] 每个线程算自己的身份信息 ────────────────────
    int warp_id = threadIdx.x >> 5;      // [THREAD]
    int lane = threadIdx.x & 31;         // [THREAD]
    int row = blockIdx.x * BR + warp_id; // [WARP] 同 warp 共享 row
    bool active = (row < M);             // [WARP] 同 warp 共享 active

    // ──── [WARP] Q 行加载:1 个 warp 协作加载 1 行 Q ──────────
    // 这一段是 "warp 内 32 个 lane 协作处理同一个 row" 的逻辑,
    // 因为 warp 内所有 lane 的 row 相同,它们一起搬运这一行的 d 个元素到寄存器
    float q[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (active)
    {
#pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            int idx = lane + (k << 5);
            if (idx < d)
                q[k] = Q[row * d + idx]; // [WARP] 协作加载,32 lane 并行
        }
    }

    // ──── [BLOCK] Shared memory 声明,被整个 block 共享 ────────
    extern __shared__ float smem[];
    float *sK = smem;
    float *sV = smem + BN * d;
    float *sScores = smem + 2 * BN * d;
    constexpr int BN_PAD = BN + 1;

    // ──── [WARP] online softmax 状态,每 warp 一份 ────────────
    // 关键!这些状态是 per-warp 的:同 warp 内所有 lane 持有的 m/l/o 相同(用 lane 0 的有效),
    // 不同 warp 的 m/l/o 独立(因为它们处理不同的 Q 行)
    float m = -FLT_MAX;
    float l = 0.0f;
    float o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    // 但实际上 m, l 在不同 lane 间是冗余的(同 warp 内值相同),
    // o[] 是 warp 内分布式存储:lane k 持有 d 维向量的第 lane+32k 位置

    const float scale = rsqrtf((float)d); // [THREAD] 每线程独立算

    // =============================================================
    // ──── 主循环:每个 KV chunk 一次迭代 ─────────────────────────
    // =============================================================
    for (int kv_start = 0; kv_start < N; kv_start += BN)
    {
        int bn = min(BN, N - kv_start);

        // ──── [BLOCK] K, V chunk 加载,4 warp 协作 ─────────────
        // 这里循环变量 i 由 threadIdx.x 决定,128 个线程一起搬数据,
        // 所以这是 BLOCK 级协作
        for (int i = threadIdx.x; i < bn * d; i += BLOCK_SIZE)
        {
            sK[i] = K[kv_start * d + i]; // [BLOCK]
            sV[i] = V[kv_start * d + i]; // [BLOCK]
        }
        __syncthreads(); // [BLOCK] 跨 warp 同步

        // ──── [WARP] 算 scores ─────────────────────────────────
        // 每个 warp 独立处理自己的 Q 行,4 个 warp 之间不通信
        if (active)
        {
            for (int j = 0; j < bn; ++j)
            {
                // [WARP] 32 lane 协作算一个 dot product
                float partial = 0.0f;
#pragma unroll
                for (int k = 0; k < 4; ++k)
                {
                    int idx = lane + (k << 5);
                    if (idx < d)
                        partial += q[k] * sK[j * d + idx]; // [THREAD] 每 lane 独立乘加
                }
                partial = warpReduceSum(partial); // [WARP] 5 步 shuffle 归约
                if (lane == 0)                    // [WARP] 仅 lane 0 写出
                    sScores[warp_id * BN_PAD + j] = partial * scale;
            }
        }
        __syncthreads(); // [BLOCK] 严格说这里不需要—— sScores 只被同 warp 读,
                         // 但保留 sync 是为了和下面 K,V 的下一轮加载同步

        // ──── [WARP] Online softmax 更新 + 输出累加 ───────────
        // 每个 warp 独立处理自己 Q 行的 softmax 状态
        if (active)
        {
            // ──── [WARP] 求 chunk 内 max ────────────────────────
            // 严格说这里 32 个 lane 都算了一遍,结果相同,是冗余计算,
            // 但保持 warp 内所有 lane 持有相同 chunk_max(否则后面就 diverge)
            float chunk_max = -FLT_MAX;
            for (int j = 0; j < bn; ++j)
                chunk_max = fmaxf(chunk_max, sScores[warp_id * BN_PAD + j]);

            // ──── [WARP] m, l 更新(同 warp 内所有 lane 持有相同值) ────
            float m_new = fmaxf(m, chunk_max);
            float alpha = __expf(m - m_new);

            // ──── [WARP] 缩放旧 l 和 o ─────────────────────────
            l *= alpha; // [WARP] 同 warp 所有 lane 同步缩放
#pragma unroll
            for (int k = 0; k < 4; ++k)
                o[k] *= alpha; // [THREAD] 每 lane 各自缩放自己负责的部分

            // ──── [WARP] 累加这个 chunk 的贡献 ─────────────────
            for (int j = 0; j < bn; ++j)
            {
                // [WARP] 同 warp 所有 lane 读同一个 sScores → broadcast,无 conflict
                float p = __expf(sScores[warp_id * BN_PAD + j] - m_new);
                l += p; // [WARP] 同步更新 l
#pragma unroll
                for (int k = 0; k < 4; ++k)
                {
                    int idx = lane + (k << 5);
                    if (idx < d)
                        o[k] += p * sV[j * d + idx]; // [THREAD] 每 lane 累加自己那段
                }
            }
            m = m_new; // [WARP] 同步更新 m
        }
        __syncthreads(); // [BLOCK] 准备下一轮 K, V 加载
    }

    // ──── [WARP] 最终归一化并写出 ─────────────────────────────
    if (active)
    {
        float inv_l = __frcp_rn(l); // [WARP] 算一次,所有 lane 拿到
#pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            int idx = lane + (k << 5);
            if (idx < d)
                O[row * d + idx] = o[k] * inv_l; // [WARP] 32 lane 协作写出 d 维输出
        }
    }
}