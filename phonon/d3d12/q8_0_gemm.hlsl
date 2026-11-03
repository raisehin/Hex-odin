// Q8_0 x Q8_0 GEMM for the phonon encoder, D3D12 compute.
//
// Weights arrive in a split layout that gemm.odin prepares on upload:
//   wq : [N][K] int8, the 32 quant bytes of each Q8_0 block, scale removed
//   wd : [N][blocks] f32, each block's scale
// Keeping the scales out of the row makes every weight access 4-byte aligned.
//
// Mapping: one warp owns one output row, and the 32 lanes cover 4
// consecutive Q8_0 blocks at a time. Lane L owns uint (L%8) of block
// (B + L/8), so the warp's 32 weight loads hit 128 contiguous bytes of the
// row and every fetched cache line is fully useful. (The previous loop,
// b = lane; b += 32, put consecutive lanes 32 bytes apart, so each 32-byte
// sector delivered only 4 useful bytes.) Each lane dots its piece against
// its FRAMES_PER_WARP frames, scales by the block scales, and the pieces
// are combined with WaveActiveSum; the low lane writes the F frames.
//
// Activations are phonon.quantize_acts output: int8 rows plus f32 block scales,
// already f16-rounded on the CPU, so both sides accumulate the same int32
// block products.

#define WARPS_PER_GROUP 4
// 16 frames per warp halves the number of frame groups, and every group
// re-reads the whole weight matrix: at T=132 that is 9 passes instead of 17.
// The GEMM is weight-stream bound (gpupeak measures 811 GFLOPS of dot4 issue
// against 75 achieved), so cutting weight traffic is the lever. facc[16] is
// 16 VGPRs, affordable. Accumulation order is untouched: each frame keeps its
// own facc slot and the same WaveActiveSum.
#define FRAMES_PER_WARP 8

cbuffer Params : register(b0)
{
    uint N;           // output rows (ne1)
    uint K;           // input columns (ne0)
    uint T;           // frames
    uint has_bias;
    uint num_blocks;  // K / 32
    uint pad0;
    uint pad1;
    uint pad2;
};

ByteAddressBuffer wq   : register(t0);  // [N][K] int8
ByteAddressBuffer wd   : register(t1);  // [N][num_blocks] f32
ByteAddressBuffer aq   : register(t2);  // [T][K] int8
ByteAddressBuffer ad   : register(t3);  // [T][num_blocks] f32
ByteAddressBuffer bbuf : register(t4);  // [N] f32
RWByteAddressBuffer obuf : register(u0);  // [T][N] f32

[numthreads(WARPS_PER_GROUP * 32, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid3 : SV_GroupThreadID)
{
    uint warp = tid3.x / 32;
    uint lane = tid3.x % 32;
    uint o = gid.x * WARPS_PER_GROUP + warp;
    if (o >= N)
        return;

    uint f0 = gid.y * FRAMES_PER_WARP;
    // Fully out-of-range groups write nothing (stores below are predicated
    // on f0+f<T with use=0 zeroing every accumulation), so skip all traffic.
    if (f0 >= T)
        return;

    float facc[FRAMES_PER_WARP];
    [unroll]
    for (uint f = 0; f < FRAMES_PER_WARP; ++f)
        facc[f] = 0.0f;

    uint wrow = o * K;
    uint drow = o * num_blocks;

    uint blk  = lane / 8; // 0..3: which of the 4 blocks in this group
    uint uidx = lane % 8; // which uint inside that block

    for (uint B = 0; B < num_blocks; B += 4)
    {
        // Clamp the tail when num_blocks is not a multiple of 4:
        // out-of-range lanes reload the last block and are scaled by
        // zero via `use`. (D3D out-of-bounds loads return zero anyway.)
        uint bb = min(B + blk, num_blocks - 1);
        uint tail = (B + blk < num_blocks) ? 1u : 0u;
        uint wv = wq.Load<uint>(wrow + bb * 32 + uidx * 4);
        float dw = wd.Load<float>((drow + bb) * 4);

        [unroll]
        for (uint f = 0; f < FRAMES_PER_WARP; ++f)
        {
            // Clamp instead of branching: out-of-range frames reuse the
            // last real frame's activations and are scaled by zero below.
            uint fg = min(f0 + f, T - 1);
            uint qv = aq.Load<uint>(fg * K + bb * 32 + uidx * 4);
            float da = ad.Load<float>((fg * num_blocks + bb) * 4);
            float use = (tail != 0u && f0 + f < T) ? 1.0f : 0.0f;
            // NOTE: the accumulator must stay a variable: this repo's dxc
            // (WinSDK 10.0.19041.0) validator rejects dot4add_i8packed with
            // a literal 0 third argument ("Invalid record").
            int acc = 0;
            acc = dot4add_i8packed(wv, qv, acc);
            facc[f] += (float)acc * dw * da * use;
        }
    }

    [unroll]
    for (uint f = 0; f < FRAMES_PER_WARP; ++f)
        facc[f] = WaveActiveSum(facc[f]);

    if (lane != 0)
        return;

    float bv = has_bias != 0 ? bbuf.Load<float>(o * 4) : 0.0f;
    [unroll]
    for (uint f = 0; f < FRAMES_PER_WARP; ++f)
    {
        if (f0 + f < T)
            obuf.Store((f0 + f) * N * 4 + o * 4, asuint(facc[f] + bv));
    }
}
