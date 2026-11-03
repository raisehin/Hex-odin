// Compute-only peak probe for the D3D12 path: eight independent
// dot4add_i8packed chains over register operands, no memory traffic in the
// loop. Used to tell "the GEMM kernel is inefficient" apart from "this iGPU's
// int8 dot throughput is the ceiling".

cbuffer Params : register(b0)
{
    uint iters;
    uint pad0;
    uint pad1;
    uint pad2;
    uint pad3;
    uint pad4;
    uint pad5;
    uint pad6;
};

RWByteAddressBuffer obuf : register(u0);

[numthreads(256, 1, 1)]
void main(uint3 tid3 : SV_DispatchThreadID)
{
    uint x = tid3.x * 2654435761u + 1u;
    int a0 = int(x);
    int a1 = int(x ^ 0x01010101u);
    int a2 = int(x ^ 0x02020202u);
    int a3 = int(x ^ 0x03030303u);
    int a4 = int(x ^ 0x04040404u);
    int a5 = int(x ^ 0x05050505u);
    int a6 = int(x ^ 0x06060606u);
    int a7 = int(x ^ 0x07070707u);

    uint v0 = (x | 0x01010101u) ^ 0x7F7F7F7Fu;
    uint v1 = (x | 0x80808080u) ^ 0x01010101u;

    for (uint i = 0; i < iters; ++i)
    {
        a0 = dot4add_i8packed(v0, v1, a0);
        a1 = dot4add_i8packed(v1, v0, a1);
        a2 = dot4add_i8packed(v0, v0, a2);
        a3 = dot4add_i8packed(v1, v1, a3);
        a4 = dot4add_i8packed(v0 ^ i, v1, a4);
        a5 = dot4add_i8packed(v1 ^ i, v0, a5);
        a6 = dot4add_i8packed(v0, v1 ^ i, a6);
        a7 = dot4add_i8packed(v1, v0 ^ i, a7);
    }

    obuf.Store(tid3.x * 4, asuint(a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7));
}
