#include "out-prod.cuh"

#include <cstdint>

// Custom out_prod kernel — replaces cuBLAS SGEMM entirely to avoid the
// ldb validation bug that some cuBLAS versions have with CUBLAS_OP_T.
// dst[i,j] = sum_k src0[i,k] * src1[j,k]   (column-major throughout)
static __global__ void out_prod_f32(const float * __restrict__ src0,
                                    const float * __restrict__ src1,
                                    float * __restrict__ dst,
                                    const int ne0, const int ne1, const int ne01,
                                    const int64_t s0_col, const int64_t s1_col,
                                    const int64_t d_col, const bool src1_T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= ne0 || j >= ne1) return;

    float sum = 0.0f;
    if (src1_T) {
        // src1 transposed: element (j,k) is at j * s1_col + k
        for (int k = 0; k < ne01; k++) {
            sum += src0[i + k * s0_col] * src1[j * s1_col + k];
        }
    } else {
        // src1 not transposed: element (j,k) is at j + k * s1_col
        for (int k = 0; k < ne01; k++) {
            sum += src0[i + k * s0_col] * src1[j + k * s1_col];
        }
    }
    dst[i + j * d_col] = sum;
}

void ggml_cuda_out_prod(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(ne01 == ne11);
    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);

    GGML_ASSERT(ne2 % src0->ne[2] == 0);
    GGML_ASSERT(ne3 % src0->ne[3] == 0);

    GGML_ASSERT(ne2 == src1->ne[2]);
    GGML_ASSERT(ne3 == src1->ne[3]);

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    cudaStream_t stream = ctx.stream();

    const int64_t s0_col = nb01 / sizeof(float);
    const int64_t d_col  = nb1  / sizeof(float);

    const bool src1_T = ggml_is_transposed(src1);
    const int64_t s1_col = (src1_T ? nb10 : nb11) / sizeof(float);
    GGML_ASSERT(          (src1_T ? nb11 : nb10) == sizeof(float));

    const size_t s02 = nb02 / sizeof(float);
    const size_t s03 = nb03 / sizeof(float);
    const size_t s12 = nb12 / sizeof(float);
    const size_t s13 = nb13 / sizeof(float);
    const size_t s2  = nb2  / sizeof(float);
    const size_t s3  = nb3  / sizeof(float);

    const int64_t dps2 = ne2 / ne02;
    const int64_t dps3 = ne3 / ne03;

    const dim3 block(16, 16);
    const dim3 grid(((int)ne0 + 15) / 16, ((int)ne1 + 15) / 16);

    for (int64_t i3 = 0; i3 < ne3; ++i3) {
        for (int64_t i2 = 0; i2 < ne2; ++i2) {
            out_prod_f32<<<grid, block, 0, stream>>>(
                src0_d + (i3/dps3)*s03 + (i2/dps2)*s02,
                src1_d +  i3      *s13 +  i2      *s12,
                dst_d  +  i3      *s3  +  i2      *s2,
                (int)ne0, (int)ne1, (int)ne01,
                s0_col, s1_col, d_col, src1_T);
        }
    }
}
