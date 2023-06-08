from statistics import mean, stdev
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

kShowFig1 = True

kShowFig2 = True

kMalloc = "TIME: malloc"
kMallocM = True

kMmcopy = "TIME: mmcopy"
kMmcpoyM = True

kPointNum = "find points num"
kPointNumM = True

kValidPillars = "find pillar_num"
kValidPillarsM = True


kGenerateVoxels = "TIME: generateVoxels"
kGenerateVoxelsM = True


kGenerateFeatures = "TIME: generateFeatures"
kGenerateFeaturesM = True

kTotalPreprocess = "total preprocess"
kTotalPreprocessM = True

kDoInfer = "TIME: doinfer"
kDoInferM = True

kPostProcess = "TIME: doPostprocessCuda"
kPostProcessM = True
kRunDetectionSum = "TIME: pointpillar"
kRunDetectionSumM = True


kLevel1 = [kMalloc, kMmcopy, kPointNum, kGenerateVoxels, kValidPillars, kTotalPreprocess, kGenerateFeatures, kDoInfer,
           kPostProcess, kRunDetectionSum]
kLevel1PlotMask = [kMallocM, kMmcpoyM, kPointNumM, kGenerateVoxelsM, kValidPillarsM, kTotalPreprocessM, kGenerateFeaturesM, kDoInferM,
                   kPostProcessM, kRunDetectionSumM]
kLevel1LastEle = [kPointNum, kValidPillars]


class InferenceInfo:
    def __init__(self) -> None:
        self.level1 = [[] for i in range(len(kLevel1))]

    def __str__(self) -> str:
        res = "InferenceInfo: \nlevel1 has:"
        for i, name in enumerate(kLevel1):
            res += " " + name + " " + str(len(self.level1[i]))
        return res

    def find_info(self, line: str) -> None:
        eles = line.strip().split(" ")
        found = False
        for i, val in enumerate(kLevel1):
            if line.find(val) > -1:
                if val in kLevel1LastEle:
                    self.level1[i].append(float(eles[-1]))
                else:
                    self.level1[i].append(float(eles[-2]))
                found = True
                break

    def plot(self, version: str, platform: str) -> None:
        malloc_time = self.level1[kLevel1.index(kMalloc)]
        mmcopy_time = self.level1[kLevel1.index(kMmcopy)]
        total = self.level1[kLevel1.index(kRunDetectionSum)]
        assert(len(malloc_time) == len(mmcopy_time))
        assert(len(malloc_time) == len(total))
        for i in range(len(malloc_time)):
            total[i] += (malloc_time[i] + mmcopy_time[i])

        generate_voxel_time = self.level1[kLevel1.index(kGenerateVoxels)]
        generate_feat_time = self.level1[kLevel1.index(kGenerateFeatures)]
        assert(len(malloc_time) == len(generate_voxel_time))
        assert(len(malloc_time) == len(generate_feat_time))

        total_preprocess = self.level1[kLevel1.index(kTotalPreprocess)]
        for i in range(len(malloc_time)):
            t = malloc_time[i] + mmcopy_time[i] + \
                generate_feat_time[i] + generate_voxel_time[i]
            total_preprocess.append(t)

        if kShowFig1:
            fig = plt.figure(figsize=(19, 10), dpi=100)
            gs = GridSpec(5, 1, figure=fig)
            ax1 = fig.add_subplot(gs[0:3, 0])
            ax2 = fig.add_subplot(gs[3:4, 0])
            ax3 = fig.add_subplot(gs[4:5, 0])
            lengends = []
            for i, val in enumerate(kLevel1):
                if (val not in kLevel1LastEle) and (kLevel1PlotMask[i]):
                    lengends.append(val + " mean: {:.3f} max: {:.3f} min: {:.3f} stdver: {:.3f}".format(
                        mean(self.level1[i]), max(self.level1[i]), min(self.level1[i]), stdev(self.level1[i])))
                    ax1.plot(self.level1[i])
            ax1.legend(lengends)
            ax1.set_title("each step consume time(ms) per frame")
            ax1.set_xlabel("frame")
            ax1.set_ylabel("consume time(ms)")
            ax1.grid()

            idx = kLevel1.index(kPointNum)
            ax2.plot(self.level1[idx])
            ax2.set_title("point number per frame")
            ax2.set_xlabel("frame")
            ax2.set_ylabel("point number")
            ax2.grid()
            ax2.legend([kPointNum])

            idx = kLevel1.index(kValidPillars)
            ax3.plot(self.level1[idx])
            ax3.set_title("pillar number per frame")
            ax3.set_xlabel("frame")
            ax3.set_ylabel("pillar number")
            ax3.grid()
            ax3.legend([kValidPillars])
            plt.subplots_adjust(left=0.05, bottom=0.05,
                                right=0.95, top=0.935, hspace=0.40)
            plt.suptitle(
                version + " version tensorRT model runtime analysis on " + platform + " platform")

        # Show all ploted figure.
        # plt.show()
        plt.savefig("pointpillar_run_analysis_result.jpg")


def Run(file_name: str, info: InferenceInfo):
    with open(file_name) as f:
        for l in f.readlines():
            info.find_info(l)


if __name__ == "__main__":
    # file_name = '/home/guangtong/workspace/PointPillars/CUDA-PointPillars/test_analyze/CUDA_PointPillars_xavier_run.log'
    # file_name = '/home/guangtong/workspace/PointPillars/CUDA-PointPillars/test_analyze/xavier_mallocmanaged_all_log.log'
    # file_name = '/home/guangtong/workspace/PointPillars/CUDA-PointPillars/test_analyze/xavier_malloc_all_log.log'
    # file_name = '/home/guangtong/workspace/PointPillars/CUDA-PointPillars/test_analyze/xavier_malloc_run_apollo_loading.log'
    # file_name = '/home/guangtong/workspace/PointPillars/CUDA-PointPillars/test_analyze/xavier_malloc_run_cpu_loading.log'
    file_name = './PointPillars_x86_run.log'
    
    inference_info = InferenceInfo()
    Run(file_name, inference_info)
    print(inference_info)
    inference_info.plot(
        'CUDA_PointPillars FP16 os_loading malloc api', "x86_3060")
