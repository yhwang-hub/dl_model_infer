#include "post_process.h"

namespace ai
{
    namespace postprocess
    {
        float box_iou_cpu(
            float aleft, float atop, float aright, float abottom,
            float bleft, float btop, float bright, float bbottom)
        {
            float cleft = std::max(aleft, bleft);
            float ctop = std::max(atop, btop);
            float cright = std::min(aright, bright);
            float cbottom = std::min(abottom, bbottom);

            float c_area = std::max(cright - cleft, 0.0f) * std::max(cbottom - ctop, 0.0f);
            if (c_area == 0.0f)
                return 0.0f;

            float a_area = std::max(0.0f, aright - aleft) * std::max(0.0f, abottom - atop);
            float b_area = std::max(0.0f, bright - bleft) * std::max(0.0f, bbottom - btop);
            return c_area / (a_area + b_area - c_area);
        }

        void fast_nms_cpu(float* bboxes, float threshold, int max_objects, int NUM_BOX_ELEMENT)
        {
            int count = std::min((int)*bboxes, max_objects);
            for (int position = 0; position < max_objects; position++)
            {
                if (position >= count)
                    return;

                float* pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT;
                for (int i = 0; i < count; i++)
                {
                    float* pitem = bboxes + 1 + i * NUM_BOX_ELEMENT;
                    if (i == position || pcurrent[5] != pitem[5])
                        continue;

                    if (pitem[4] >= pcurrent[4])
                    {
                        if (pitem[4] == pcurrent[4] && i < position)
                            continue;

                        float iou = box_iou_cpu(
                            pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
                            pitem[0], pitem[1], pitem[2], pitem[3]
                        );

                        if (iou > threshold)
                        {
                            pcurrent[6] = 0;// 1=keep, 0=ignore
                        }
                    }
                }
            }
        }
    }
}