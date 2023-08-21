#ifndef _IOU3D_NMS_H
#define _IOU3D_NMS_H
/*
3D IoU Calculation and Rotated NMS(modified from 2D NMS written by others)
Written by Shaoshuai Shi
All Rights Reserved 2019-2022.
*/
#pragma once

#include <stdio.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <vector>
#include <iostream>

#include "../../common/cuda_utils.h"

class Iou3dNmsCuda
{
public:
  Iou3dNmsCuda(const int head_x_size,
               const int head_y_size,
               const float nms_overlap_thresh);
  ~Iou3dNmsCuda() = default;

  int DoIou3dNms(const int box_num_pre,
                 const float* res_box,
                 const int* res_sorted_indices,
                 long* host_keep_data);

private:
  const int head_x_size_;
  const int head_y_size_;
  const float nms_overlap_thresh_;
};

// struct Box {
//   float x;
//   float y;
//   float z;
//   float l;
//   float w;
//   float h;
//   float r;
//   float vx = 0.0f;  // optional
//   float vy = 0.0f;  // optional
//   float score;
//   int label;
//   bool is_drop;  // for nms
// };

// struct bevBox
// {
//     float x, y, z, l, w, h, r;
//     float vx = 0.0f;  // optional
//     float vy = 0.0f;  // optional
//     float score;
//     int label;
//     bool is_drop;  // for nms
//     bevBox() = default;
//     bevBox(float x, float y, float z, float l, float w, float h, float r,
//             float vx, float vy, float score, int label, bool is_drop)
//             : x(x), y(y), z(z), l(l), w(w), h(h), r(r),
//             vx(vx), vy(vy), score(score), label(label), is_drop(is_drop){}
// };

#endif