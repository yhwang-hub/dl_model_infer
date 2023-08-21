#include "pre_process.h"
namespace ai
{
    namespace preprocess
    {
        // same to opencv
        // reference: https://github.com/opencv/opencv/blob/24fcb7f8131f707717a9f1871b17d95e7cf519ee/modules/imgproc/src/resize.cpp
        // reference: https://github.com/openppl-public/ppl.cv/blob/04ef4ca48262601b99f1bb918dcd005311f331da/src/ppl/cv/cuda/resize.cu
        __global__ void resize_bilinear_and_normalize_kernel(
            uint8_t *src, int src_line_size, int src_width, int src_height, float *dst, int dst_width, int dst_height,
            float sx, float sy, Norm norm, int edge)
        {
            int position = blockDim.x * blockIdx.x + threadIdx.x;
            if (position >= edge)
                return;

            int dx = position % dst_width;
            int dy = position / dst_width;
            float src_x = (dx + 0.5f) * sx - 0.5f;
            float src_y = (dy + 0.5f) * sy - 0.5f;
            float c0, c1, c2;

            int x_low = floorf(src_x);
            int y_low = floorf(src_y);
            int y_high = limit(y_low + 1, 0, src_height - 1);
            int x_high = limit(x_low + 1, 0, src_width - 1);
            y_low = limit(y_low, 0, src_height - 1);
            x_low = limit(x_low, 0, src_width - 1);

            int ly = rintf((src_y - y_low) * INTER_RESIZE_COEF_SCALE);
            int lx = rintf((src_x - x_low) * INTER_RESIZE_COEF_SCALE);
            int hy = INTER_RESIZE_COEF_SCALE - ly;
            int hx = INTER_RESIZE_COEF_SCALE - lx;
            int w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
            float *pdst = dst + dy * dst_width + dx * 3;
            uint8_t *v1 = src + y_low * src_line_size + x_low * 3;
            uint8_t *v2 = src + y_low * src_line_size + x_high * 3;
            uint8_t *v3 = src + y_high * src_line_size + x_low * 3;
            uint8_t *v4 = src + y_high * src_line_size + x_high * 3;

            c0 = resize_cast(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0]);
            c1 = resize_cast(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1]);
            c2 = resize_cast(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2]);

            if (norm.channel_type == ChannelType::RGB)
            {
                float t = c2;
                c2 = c0;
                c0 = t;
            }

            if (norm.type == NormType::MeanStd)
            {
                c0 = (c0 * norm.alpha - norm.mean[0]) / norm.std[0];
                c1 = (c1 * norm.alpha - norm.mean[1]) / norm.std[1];
                c2 = (c2 * norm.alpha - norm.mean[2]) / norm.std[2];
            }
            else if (norm.type == NormType::AlphaBeta)
            {
                c0 = c0 * norm.alpha + norm.beta;
                c1 = c1 * norm.alpha + norm.beta;
                c2 = c2 * norm.alpha + norm.beta;
            }

            int area = dst_width * dst_height;
            float *pdst_c0 = dst + dy * dst_width + dx;
            float *pdst_c1 = pdst_c0 + area;
            float *pdst_c2 = pdst_c1 + area;
            *pdst_c0 = c0;
            *pdst_c1 = c1;
            *pdst_c2 = c2;
        }

        __global__ void warp_perspective_kernel(uint8_t *src, int src_line_size, int src_width, int src_height, float *dst, int dst_width, int dst_height,
                                                uint8_t const_value_st, float *warp_affine_matrix_3_3, Norm norm, int edge)
        {

            int position = blockDim.x * blockIdx.x + threadIdx.x;
            if (position >= edge)
                return;

            float m_x1 = warp_affine_matrix_3_3[0];
            float m_y1 = warp_affine_matrix_3_3[1];
            float m_z1 = warp_affine_matrix_3_3[2];

            float m_x2 = warp_affine_matrix_3_3[3];
            float m_y2 = warp_affine_matrix_3_3[4];
            float m_z2 = warp_affine_matrix_3_3[5];

            float m_x3 = warp_affine_matrix_3_3[6];
            float m_y3 = warp_affine_matrix_3_3[7];
            float m_z3 = warp_affine_matrix_3_3[8];

            int dx = position % dst_width;
            int dy = position / dst_width;

            // 原图位置
            float src_x = (m_x1 * dx + m_y1 * dy + m_z1) / (m_x3 * dx + m_y3 * dy + m_z3);
            float src_y = (m_x2 * dx + m_y2 * dy + m_z2) / (m_x3 * dx + m_y3 * dy + m_z3);
            float c0, c1, c2;

            if (src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height)
            {
                // out of range
                c0 = const_value_st;
                c1 = const_value_st;
                c2 = const_value_st;
            }
            else
            {
                int y_low = floorf(src_y);
                int x_low = floorf(src_x);
                int y_high = y_low + 1;
                int x_high = x_low + 1;

                uint8_t const_value[] = {const_value_st, const_value_st, const_value_st};
                float ly = src_y - y_low;
                float lx = src_x - x_low;
                float hy = 1 - ly;
                float hx = 1 - lx;
                float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
                uint8_t *v1 = const_value;
                uint8_t *v2 = const_value;
                uint8_t *v3 = const_value;
                uint8_t *v4 = const_value;
                if (y_low >= 0)
                {
                    if (x_low >= 0)
                        v1 = src + y_low * src_line_size + x_low * 3;

                    if (x_high < src_width)
                        v2 = src + y_low * src_line_size + x_high * 3;
                }

                if (y_high < src_height)
                {
                    if (x_low >= 0)
                        v3 = src + y_high * src_line_size + x_low * 3;

                    if (x_high < src_width)
                        v4 = src + y_high * src_line_size + x_high * 3;
                }

                // same to opencv
                c0 = floorf(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f);
                c1 = floorf(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1] + 0.5f);
                c2 = floorf(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2] + 0.5f);
            }

            if (norm.channel_type == ChannelType::RGB)
            {
                float t = c2;
                c2 = c0;
                c0 = t;
            }

            if (norm.type == NormType::MeanStd)
            {
                c0 = (c0 * norm.alpha - norm.mean[0]) / norm.std[0];
                c1 = (c1 * norm.alpha - norm.mean[1]) / norm.std[1];
                c2 = (c2 * norm.alpha - norm.mean[2]) / norm.std[2];
            }
            else if (norm.type == NormType::AlphaBeta)
            {
                c0 = c0 * norm.alpha + norm.beta;
                c1 = c1 * norm.alpha + norm.beta;
                c2 = c2 * norm.alpha + norm.beta;
            }

            int area = dst_width * dst_height;
            float *pdst_c0 = dst + dy * dst_width + dx;
            float *pdst_c1 = pdst_c0 + area;
            float *pdst_c2 = pdst_c1 + area;
            *pdst_c0 = c0;
            *pdst_c1 = c1;
            *pdst_c2 = c2;
        }

        __global__ void warp_affine_bilinear_and_normalize_plane_kernel(uint8_t *src, int src_line_size, int src_width, int src_height, float *dst, int dst_width, int dst_height,
                                                                        uint8_t const_value_st, float *warp_affine_matrix_2_3, Norm norm, int edge)
        {

            int position = blockDim.x * blockIdx.x + threadIdx.x;
            if (position >= edge)
                return;

            float m_x1 = warp_affine_matrix_2_3[0];
            float m_y1 = warp_affine_matrix_2_3[1];
            float m_z1 = warp_affine_matrix_2_3[2];
            float m_x2 = warp_affine_matrix_2_3[3];
            float m_y2 = warp_affine_matrix_2_3[4];
            float m_z2 = warp_affine_matrix_2_3[5];

            int dx = position % dst_width;
            int dy = position / dst_width;
            float src_x = m_x1 * dx + m_y1 * dy + m_z1;
            float src_y = m_x2 * dx + m_y2 * dy + m_z2;
            float c0, c1, c2;

            if (src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height)
            {
                // out of range
                c0 = const_value_st;
                c1 = const_value_st;
                c2 = const_value_st;
            }
            else
            {
                int y_low = floorf(src_y);
                int x_low = floorf(src_x);
                int y_high = y_low + 1;
                int x_high = x_low + 1;

                uint8_t const_value[] = {const_value_st, const_value_st, const_value_st};
                float ly = src_y - y_low;
                float lx = src_x - x_low;
                float hy = 1 - ly;
                float hx = 1 - lx;
                float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
                uint8_t *v1 = const_value;
                uint8_t *v2 = const_value;
                uint8_t *v3 = const_value;
                uint8_t *v4 = const_value;
                if (y_low >= 0)
                {
                    if (x_low >= 0)
                        v1 = src + y_low * src_line_size + x_low * 3;

                    if (x_high < src_width)
                        v2 = src + y_low * src_line_size + x_high * 3;
                }

                if (y_high < src_height)
                {
                    if (x_low >= 0)
                        v3 = src + y_high * src_line_size + x_low * 3;

                    if (x_high < src_width)
                        v4 = src + y_high * src_line_size + x_high * 3;
                }

                // same to opencv
                c0 = floorf(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f);
                c1 = floorf(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1] + 0.5f);
                c2 = floorf(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2] + 0.5f);
            }

            if (norm.channel_type == ChannelType::RGB)
            {
                float t = c2;
                c2 = c0;
                c0 = t;
            }

            if (norm.type == NormType::MeanStd)
            {
                c0 = (c0 * norm.alpha - norm.mean[0]) / norm.std[0];
                c1 = (c1 * norm.alpha - norm.mean[1]) / norm.std[1];
                c2 = (c2 * norm.alpha - norm.mean[2]) / norm.std[2];
            }
            else if (norm.type == NormType::AlphaBeta)
            {
                c0 = c0 * norm.alpha + norm.beta;
                c1 = c1 * norm.alpha + norm.beta;
                c2 = c2 * norm.alpha + norm.beta;
            }

            int area = dst_width * dst_height;
            float *pdst_c0 = dst + dy * dst_width + dx;
            float *pdst_c1 = pdst_c0 + area;
            float *pdst_c2 = pdst_c1 + area;
            *pdst_c0 = c0;
            *pdst_c1 = c1;
            *pdst_c2 = c2;
        }

        __global__ void warp_affine_bilinear_and_normalize_focus_kernel(uint8_t *src, int src_line_size, int src_width, int src_height, float *dst, int dst_width, int dst_height,
                                                                        uint8_t const_value_st, float *warp_affine_matrix_1_3, Norm norm, int edge)
        {

            int position = blockDim.x * blockIdx.x + threadIdx.x;
            if (position >= edge)
                return;

            float m_k = *warp_affine_matrix_1_3++;
            float m_b0 = *warp_affine_matrix_1_3++;
            float m_b1 = *warp_affine_matrix_1_3++;

            int dx = position % dst_width;
            int dy = position / dst_width;
            float src_x = m_k * dx + m_b0;
            float src_y = m_k * dy + m_b1;
            float c0, c1, c2;

            if (src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height)
            {
                // out of range
                c0 = const_value_st;
                c1 = const_value_st;
                c2 = const_value_st;
            }
            else
            {
                int y_low = floorf(src_y);
                int x_low = floorf(src_x);
                int y_high = y_low + 1;
                int x_high = x_low + 1;

                uint8_t const_value[] = {const_value_st, const_value_st, const_value_st};
                float ly = src_y - y_low;
                float lx = src_x - x_low;
                float hy = 1 - ly;
                float hx = 1 - lx;
                float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
                uint8_t *v1 = const_value;
                uint8_t *v2 = const_value;
                uint8_t *v3 = const_value;
                uint8_t *v4 = const_value;
                if (y_low >= 0)
                {
                    if (x_low >= 0)
                        v1 = src + y_low * src_line_size + x_low * 3;

                    if (x_high < src_width)
                        v2 = src + y_low * src_line_size + x_high * 3;
                }

                if (y_high < src_height)
                {
                    if (x_low >= 0)
                        v3 = src + y_high * src_line_size + x_low * 3;

                    if (x_high < src_width)
                        v4 = src + y_high * src_line_size + x_high * 3;
                }

                // same to opencv
                c0 = floorf(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f);
                c1 = floorf(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1] + 0.5f);
                c2 = floorf(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2] + 0.5f);
            }

            if (norm.channel_type == ChannelType::RGB)
            {
                float t = c2;
                c2 = c0;
                c0 = t;
            }

            if (norm.type == NormType::MeanStd)
            {
                c0 = (c0 * norm.alpha - norm.mean[0]) / norm.std[0];
                c1 = (c1 * norm.alpha - norm.mean[1]) / norm.std[1];
                c2 = (c2 * norm.alpha - norm.mean[2]) / norm.std[2];
            }
            else if (norm.type == NormType::AlphaBeta)
            {
                c0 = c0 * norm.alpha + norm.beta;
                c1 = c1 * norm.alpha + norm.beta;
                c2 = c2 * norm.alpha + norm.beta;
            }

            int after_focus_width = dst_width / 2;
            int after_focus_height = dst_height / 2;
            int fdx = dx / 2;
            int fdy = dy / 2;
            int fc = ((dx % 2) << 1) | (dy % 2);

            /**
             *   x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]
             *    4                     fc
             *    3                     [0, 1, 2]
             *    after_focus_height    fdy
             *    after_focus_width     fdx
             *    左乘右加
             **/

            float *pdst_c0 = dst + ((fc * 3 + 0) * after_focus_height + fdy) * after_focus_width + fdx;
            float *pdst_c1 = dst + ((fc * 3 + 1) * after_focus_height + fdy) * after_focus_width + fdx;
            float *pdst_c2 = dst + ((fc * 3 + 2) * after_focus_height + fdy) * after_focus_width + fdx;

            *pdst_c0 = c0;
            *pdst_c1 = c1;
            *pdst_c2 = c2;
        }

        __global__ void normalize_feature_kernel(float *feature_array, int num_feature, int feature_length, int edge)
        {

            /*
            &   1 gz         bi.z   0
            *   1 gy         bi.y   0
            *   N NF         bi.x   ~
            *   1 1          ti.z   0
            *   F FL / 32    ti.y   ~
            *   Q 32         ti.x   ~
            */

            int position = (blockIdx.x * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
            if (position >= edge)
                return;

            extern __shared__ float l2_norm[];

            int irow = position / feature_length;
            int icol = position % feature_length;

            if (icol == 0)
                l2_norm[irow] = 0;

            __syncthreads();

            float value = feature_array[position];
            atomicAdd(l2_norm + irow, value * value);

            __syncthreads();
            if (icol == 0)
                l2_norm[irow] = sqrt(l2_norm[irow]);

            __syncthreads();
            feature_array[position] = value / l2_norm[irow];
        }

        static __device__ uint8_t cast(float value)
        {
            return value < 0 ? 0 : (value > 255 ? 255 : value);
        }

        static __global__ void convert_nv12_to_bgr_kernel(const uint8_t *y, const uint8_t *uv, int width, int height, int linesize, uint8_t *dst_bgr, int edge)
        {

            int position = blockDim.x * blockIdx.x + threadIdx.x;
            if (position >= edge)
                return;

            int ox = position % width;
            int oy = position / width;
            const uint8_t &yvalue = y[oy * linesize + ox];
            int offset_uv = (oy >> 1) * linesize + (ox & 0xFFFFFFFE);
            const uint8_t &u = uv[offset_uv + 0];
            const uint8_t &v = uv[offset_uv + 1];
            dst_bgr[position * 3 + 0] = 1.164f * (yvalue - 16.0f) + 2.018f * (u - 128.0f);
            dst_bgr[position * 3 + 1] = 1.164f * (yvalue - 16.0f) - 0.813f * (v - 128.0f) - 0.391f * (u - 128.0f);
            dst_bgr[position * 3 + 2] = 1.164f * (yvalue - 16.0f) + 1.596f * (v - 128.0f);
        }
        /******************************************************************************************************************************************/
        /* 下面是函数的实现，主要是用来调用kernel函数 */
        void convert_nv12_to_bgr_invoke(
            const uint8_t *y, const uint8_t *uv, int width, int height, int linesize, uint8_t *dst, cudaStream_t stream)
        {

            int total = width * height;
            dim3 grid = CUDATools::grid_dims(total);
            dim3 block = CUDATools::block_dims(total);

            checkCudaKernel(convert_nv12_to_bgr_kernel<<<grid, block, 0, stream>>>(
                y, uv, width, height, linesize,
                dst, total));
        }

        void warp_affine_bilinear_and_normalize_plane(
            uint8_t *src, int src_line_size, int src_width, int src_height,
            float *dst, int dst_width, int dst_height,
            float *matrix_2_3, uint8_t const_value, const Norm &norm,
            cudaStream_t stream)
        {

            int jobs = dst_width * dst_height;
            auto grid = CUDATools::grid_dims(jobs);
            auto block = CUDATools::block_dims(jobs);

            checkCudaKernel(warp_affine_bilinear_and_normalize_plane_kernel<<<grid, block, 0, stream>>>(
                src, src_line_size,
                src_width, src_height, dst,
                dst_width, dst_height, const_value, matrix_2_3, norm, jobs));
        }

        void warp_affine_bilinear_and_normalize_focus(
            uint8_t *src, int src_line_size, int src_width, int src_height,
            float *dst, int dst_width, int dst_height,
            float *matrix_1_3, uint8_t const_value, const Norm &norm,
            cudaStream_t stream)
        {

            int jobs = dst_width * dst_height;
            auto grid = CUDATools::grid_dims(jobs);
            auto block = CUDATools::block_dims(jobs);

            checkCudaKernel(warp_affine_bilinear_and_normalize_focus_kernel<<<grid, block, 0, stream>>>(
                src, src_line_size,
                src_width, src_height, dst,
                dst_width, dst_height, const_value, matrix_1_3, norm, jobs));
        }

        void warp_perspective(
            uint8_t *src, int src_line_size, int src_width, int src_height, float *dst, int dst_width, int dst_height,
            float *matrix_3_3, uint8_t const_value, const Norm &norm, cudaStream_t stream)
        {
            int jobs = dst_width * dst_height;
            auto grid = CUDATools::grid_dims(jobs);
            auto block = CUDATools::block_dims(jobs);

            checkCudaKernel(warp_perspective_kernel<<<grid, block, 0, stream>>>(
                src, src_line_size,
                src_width, src_height, dst,
                dst_width, dst_height, const_value, matrix_3_3, norm, jobs));
        }

        void resize_bilinear_and_normalize(
            uint8_t *src, int src_line_size, int src_width, int src_height, float *dst, int dst_width, int dst_height,
            const Norm &norm,
            cudaStream_t stream)
        {

            int jobs = dst_width * dst_height;
            auto grid = CUDATools::grid_dims(jobs);
            auto block = CUDATools::block_dims(jobs);

            checkCudaKernel(resize_bilinear_and_normalize_kernel<<<grid, block, 0, stream>>>(
                src, src_line_size,
                src_width, src_height, dst,
                dst_width, dst_height, src_width / (float)dst_width, src_height / (float)dst_height, norm, jobs));
        }

        void norm_feature(
            float *feature_array, int num_feature, int feature_length,
            cudaStream_t stream)
        {
            Assert(feature_length % 32 == 0);

            int jobs = num_feature * feature_length;
            auto grid = dim3(num_feature);
            auto block = dim3(feature_length / 32, 32);
            checkCudaKernel(normalize_feature_kernel<<<grid, block, num_feature * sizeof(float), stream>>>(
                feature_array, num_feature, feature_length, jobs));
        }



        /****************bevdet preprocess****************/
        // resize, crop, norm
        // sample : Bicubic
        __global__ void preprocess_nearest_kernel(const uchar* __restrict__ src_dev, 
                                            float* __restrict__ dst_dev, int src_row_step, 
                                            int dst_row_step, int src_img_step, int dst_img_step,
                                            int src_h, int src_w, float radio_h, float radio_w, 
                                            float offset_h, float offset_w, BboxDim mean, BboxDim std)
        {
            int i = blockIdx.x;
            int j = blockIdx.y;
            int k = threadIdx.x;

            int pX = (int) roundf((i / radio_h) + offset_h);
            int pY = (int) roundf((j / radio_w) + offset_w);
        
            if (pX < src_h && pX >= 0 && pY < src_w && pY >= 0)
            {
                int s1 = k * src_img_step + 0 * src_img_step / 3 + pX * src_row_step + pY;
                int s2 = k * src_img_step + 1 * src_img_step / 3 + pX * src_row_step + pY;
                int s3 = k * src_img_step + 2 * src_img_step / 3 + pX * src_row_step + pY;

                int d1 = k * dst_img_step + 0 * dst_img_step / 3 + i * dst_row_step + j;
                int d2 = k * dst_img_step + 1 * dst_img_step / 3 + i * dst_row_step + j;
                int d3 = k * dst_img_step + 2 * dst_img_step / 3 + i * dst_row_step + j;

                *(dst_dev + d1) = ((float)*(src_dev + s1) - mean.x) / std.x;
                *(dst_dev + d2) = ((float)*(src_dev + s2) - mean.y) / std.y;
                *(dst_dev + d3) = ((float)*(src_dev + s3) - mean.z) / std.z;
            }
        }

        __device__ double Weight(double x, double a = -0.5)
        {
            if(x < 0.0){
                x = -x;
            }
            if(x <= 1.0){
                return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
            }
            else if (x < 2.0){
                return  (((x - 5.0) * x + 8.0) * x - 4.0) * a;

            }
            return 0.0;
        }

        __global__ void preprocess_bicubic_kernel(const uchar* __restrict__ src_dev,
                                            float* __restrict__ dst_dev, int src_row_step, 
                                            int dst_row_step, int src_img_step, int dst_img_step,
                                            int src_h, int src_w, float radio_h, float radio_w, 
                                            float offset_h, float offset_w, BboxDim mean, BboxDim std)
        {

            /*
            src_dev : 6 * 3 * src_h * src_w
            dst_dev : 6 * 3 * blockDim.x * blockDim.y
            src_row_step : src_w
            dst_row_step : blockDim.y
            src_img_step : src_h * src_w * 3
            dst_img_step : 3 * blockDim.x * blockDim.y
            src_h : height of source image
            src_w : width of source image
            radio_h : resize radio on height
            radio_w : resize radio on width
            offset_h : crop offset, crop_h / resize_radio_h, 在原图像上纵向自上裁剪的像素范围, crop_h表示resize后的图像纵向裁剪的范围
            offset_w : 同上
            */
            
            int i = blockIdx.x;
            int j = blockIdx.y;
            int k = threadIdx.x;
            int l = threadIdx.y;

            double pX = (i / radio_h) + offset_h;
            double pY = (j / radio_w) + offset_w;

            int u = l / 4 - 1;
            int v = l % 4 - 1;

            int src_xidx = u + (int)pX;
            if(src_xidx < 0 || src_xidx >= src_h){
                return;
            }
            int src_yidx = v + (int)pY;
            if(src_yidx < 0 || src_yidx >= src_w){
                return;
            }
            double w = Weight((double)src_xidx - pX) * Weight((double)src_yidx - pY);
            
            int s1 = k * src_img_step + 0 * src_img_step / 3 + src_xidx * src_row_step + src_yidx;
            int s2 = k * src_img_step + 1 * src_img_step / 3 + src_xidx * src_row_step + src_yidx;
            int s3 = k * src_img_step + 2 * src_img_step / 3 + src_xidx * src_row_step + src_yidx;

            int d1 = k * dst_img_step + 0 * dst_img_step / 3 + i * dst_row_step + j;
            int d2 = k * dst_img_step + 1 * dst_img_step / 3 + i * dst_row_step + j;
            int d3 = k * dst_img_step + 2 * dst_img_step / 3 + i * dst_row_step + j;

            atomicAdd(dst_dev + d1, w * ((float)*(src_dev + s1)) / std.x - mean.x / std.x / 16.0f);
            atomicAdd(dst_dev + d2, w * ((float)*(src_dev + s2)) / std.y - mean.y / std.y / 16.0f);
            atomicAdd(dst_dev + d3, w * ((float)*(src_dev + s3)) / std.z - mean.z / std.z / 16.0f);
        }


        __global__ void fill_in_kernel(float* array, float num)
        {
            // gridDim.x : h
            // gridDim.y : w
            // blockDim.x: 18
            int offset = blockIdx.x * gridDim.y * blockDim.x + blockIdx.y * blockDim.x + threadIdx.x;
            array[offset] = num;
        }

        int bevdet_preprocess(const uchar* src_imgs, float* dst_imgs, int n_img, int src_img_h,
                int src_img_w, int dst_img_h, int dst_img_w, float resize_radio_h, 
                float resize_radio_w, int crop_h, int crop_w, BboxDim mean, 
                BboxDim std, Sampler sample)
        {
            /*
            src_imgs : 6 * 3 * src_img_h * src_img_w
            dst_imgs : 6 * 3 * dst_img_h * dst_img_w
            crop_h : resize后的图像，纵向自上裁剪范围
            crop_w : 为0
            */
            int src_row_step = src_img_w;
            int dst_row_step = dst_img_w;
            int src_img_step = src_img_w * src_img_h * 3;
            int dst_img_step = dst_img_w * dst_img_h * 3;

            float offset_h = crop_h / resize_radio_h;
            float offset_w = crop_w / resize_radio_w;

            dim3 grid(dst_img_h, dst_img_w);
            dim3 block;
            if(sample == Sampler::bicubic)
            {
                printf("sampler : bicubic\n");
                block = dim3(n_img, 16);
                fill_in_kernel<<<dim3(dst_img_h, dst_img_w), dim3(n_img * 3)>>>(dst_imgs, 0.0f);

                CHECK(cudaDeviceSynchronize());
                preprocess_bicubic_kernel<<<grid, block>>>(src_imgs, dst_imgs, src_row_step, 
                                        dst_row_step, src_img_step, dst_img_step, src_img_h, 
                                        src_img_w, resize_radio_h, resize_radio_w, offset_h, 
                                                                        offset_w, mean, std);
            }
            else if(sample == Sampler::nearest)
            {
                printf("sampler : nearest\n");
                block = dim3(n_img);
                preprocess_nearest_kernel<<<grid, block>>>(src_imgs, dst_imgs, src_row_step, dst_row_step, 
                                src_img_step, dst_img_step, src_img_h, src_img_w, resize_radio_h,
                                resize_radio_w, offset_h, offset_w, mean, std);
            }

            return EXIT_SUCCESS;
        }

        __global__ void convert_RGBHWC_to_BGRCHW_kernel(uchar *input, uchar *output, 
                                                     int channels, int height, int width)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if(index < channels * height * width)
            {
                int y = index / 3 / width;
                int x = index / 3 % width;
                int c = 2 - index % 3;  // RGB to BGR

                output[c * height * width + y * width + x] = input[index];
            }
        }

        // RGBHWC to BGRCHW
        void convert_RGBHWC_to_BGRCHW(uchar *input, uchar *output, 
                                    int channels, int height, int width)
        {
            convert_RGBHWC_to_BGRCHW_kernel<<<DIVUP(channels * height * width, NUM_THREADS), NUM_THREADS>>>
                                                                    (input, output, channels, height, width);
        }
        /****************bevdet preprocess****************/
    }
}