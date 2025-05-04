#!/usr/bin/env python
import torch
import torch.nn as nn
import math
from .correlation.correlation import FunctionCorrelation  


# Global dictionaries for caching grids used for backwarping
backwarp_tenGrid = {}
backwarp_tenPartial = {}

def backwarp(tenInput, tenFlow):
    """Warps an input tensor using a flow field."""
    shape_str = str(tenFlow.shape)
    if shape_str not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, -1).repeat(1, 1, tenFlow.shape[2], 1)
        tenVer = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, -1, 1).repeat(1, 1, 1, tenFlow.shape[3])
        backwarp_tenGrid[shape_str] = torch.cat([tenHor, tenVer], 1).to(tenFlow.device)
    if shape_str not in backwarp_tenPartial:
        backwarp_tenPartial[shape_str] = tenFlow.new_ones([tenFlow.shape[0], 1, tenFlow.shape[2], tenFlow.shape[3]])
    tenFlow = torch.cat([
        tenFlow[:, 0:1, :, :] * (2.0 / (tenInput.shape[3] - 1.0)),
        tenFlow[:, 1:2, :, :] * (2.0 / (tenInput.shape[2] - 1.0))
    ], 1)
    tenInput = torch.cat([tenInput, backwarp_tenPartial[shape_str]], 1)
    tenOutput = nn.functional.grid_sample(
        input=tenInput,
        grid=(backwarp_tenGrid[shape_str] + tenFlow).permute(0, 2, 3, 1),
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )
    tenMask = tenOutput[:, -1:, :, :].clone()
    tenMask[tenMask > 0.999] = 1.0
    tenMask[tenMask < 1.0] = 0.0
    return tenOutput[:, :-1, :, :] * tenMask

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # --- Extractor ---
        # (Adjusted to match the checkpoint’s weight shapes)
        class Extractor(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.netOne = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netThr = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netFou = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netSix = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
            # end
            def forward(self, tenInput):
                tenOne = self.netOne(tenInput)
                tenTwo = self.netTwo(tenOne)
                tenThr = self.netThr(tenTwo)
                tenFou = self.netFou(tenThr)
                tenFiv = self.netFiv(tenFou)
                tenSix = self.netSix(tenFiv)
                return [tenOne, tenTwo, tenThr, tenFou, tenFiv, tenSix]
        # --- Decoder and Refiner (as in your offline implementations) ---
        class Decoder(torch.nn.Module):
            def __init__(self, intLevel):
                super().__init__()

                intPrevious = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None ][intLevel + 1]
                intCurrent = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None ][intLevel + 0]

                if intLevel < 6: self.netUpflow = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
                if intLevel < 6: self.netUpfeat = torch.nn.ConvTranspose2d(in_channels=intPrevious + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=4, stride=2, padding=1)
                if intLevel < 6: self.fltBackwarp = [ None, None, None, 5.0, 2.5, 1.25, 0.625, None ][intLevel + 1]

                self.netOne = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netThr = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128, out_channels=96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netFou = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netSix = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=3, stride=1, padding=1)
                )
            # end

            def forward(self, tenOne, tenTwo, objPrevious):
                tenFlow = None
                tenFeat = None

                if objPrevious is None:
                    tenFlow = None
                    tenFeat = None

                    tenVolume = torch.nn.functional.leaky_relu(input=FunctionCorrelation(tenOne=tenOne, tenTwo=tenTwo), negative_slope=0.1, inplace=False)

                    tenFeat = torch.cat([ tenVolume ], 1)

                elif objPrevious is not None:
                    tenFlow = self.netUpflow(objPrevious['tenFlow'])
                    tenFeat = self.netUpfeat(objPrevious['tenFeat'])

                    tenVolume = torch.nn.functional.leaky_relu(input=FunctionCorrelation(tenOne=tenOne, tenTwo=backwarp(tenInput=tenTwo, tenFlow=tenFlow * self.fltBackwarp)), negative_slope=0.1, inplace=False)

                    tenFeat = torch.cat([ tenVolume, tenOne, tenFlow, tenFeat ], 1)

                # end

                tenFeat = torch.cat([ self.netOne(tenFeat), tenFeat ], 1)
                tenFeat = torch.cat([ self.netTwo(tenFeat), tenFeat ], 1)
                tenFeat = torch.cat([ self.netThr(tenFeat), tenFeat ], 1)
                tenFeat = torch.cat([ self.netFou(tenFeat), tenFeat ], 1)
                tenFeat = torch.cat([ self.netFiv(tenFeat), tenFeat ], 1)

                tenFlow = self.netSix(tenFeat)

                return {
                    'tenFlow': tenFlow,
                    'tenFeat': tenFeat
                }
            # end
        # end
        class Refiner(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.netMain = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)
                )
            # end
            def forward(self, tenInput):
                return self.netMain(tenInput)
        self.netExtractor = Extractor()

        self.netTwo = Decoder(2)
        self.netThr = Decoder(3)
        self.netFou = Decoder(4)
        self.netFiv = Decoder(5)
        self.netSix = Decoder(6)

        self.netRefiner = Refiner()

        self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/github/pytorch-pwc/network-' + 'default' + '.pytorch', file_name='pwc-' + 'default').items() })
    # end
    def forward(self, tenOne, tenTwo):
        tenOne = self.netExtractor(tenOne)
        tenTwo = self.netExtractor(tenTwo)

        objEstimate = self.netSix(tenOne[-1], tenTwo[-1], None)
        objEstimate = self.netFiv(tenOne[-2], tenTwo[-2], objEstimate)
        objEstimate = self.netFou(tenOne[-3], tenTwo[-3], objEstimate)
        objEstimate = self.netThr(tenOne[-4], tenTwo[-4], objEstimate)
        objEstimate = self.netTwo(tenOne[-5], tenTwo[-5], objEstimate)

        return (objEstimate['tenFlow'] + self.netRefiner(objEstimate['tenFeat'])) * 20.0
    
    def estimate(self, tenOne: torch.Tensor, tenTwo: torch.Tensor) -> torch.Tensor:
        """
        Compute optical flow between two frames (C×H×W), returns 2×H×W on CPU.
        """
        assert tenOne.shape == tenTwo.shape, "Input tensors must match"
        b, c, h, w = 1, *tenOne.shape
        # add batch
        x1 = tenOne.unsqueeze(0).cuda()
        x2 = tenTwo.unsqueeze(0).cuda()

        # pad to multiple of 64
        new_w = math.ceil(w/64)*64
        new_h = math.ceil(h/64)*64
        x1p = nn.functional.interpolate(x1, size=(new_h, new_w), align_corners=False, mode='bilinear')
        x2p = nn.functional.interpolate(x2, size=(new_h, new_w), align_corners=False, mode='bilinear')

        # forward
        flow = self.forward(x1p, x2p)

        # resize flow back
        flow = nn.functional.interpolate(flow, size=(h, w), align_corners=False, mode='bilinear')
        flow[:,0,:,:] *= w / new_w
        flow[:,1,:,:] *= h / new_h

        return flow[0].cpu()
    
    # def estimate(self, tenOne: torch.Tensor, tenTwo: torch.Tensor) -> torch.Tensor:
    #     assert(tenOne.shape[1] == tenTwo.shape[1])
    #     assert(tenOne.shape[2] == tenTwo.shape[2])

    #     intWidth = tenOne.shape[2]
    #     intHeight = tenOne.shape[1]

    #     # assert(intWidth == 1024) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
    #     # assert(intHeight == 436) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
    #     # assert(intWidth == 4096) 
    #     # assert(intHeight == 1200) 

    #     tenPreprocessedOne = tenOne.cuda().view(1, 3, intHeight, intWidth)
    #     tenPreprocessedTwo = tenTwo.cuda().view(1, 3, intHeight, intWidth)

    #     intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
    #     intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

    #     tenPreprocessedOne = torch.nn.functional.interpolate(input=tenPreprocessedOne, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
    #     tenPreprocessedTwo = torch.nn.functional.interpolate(input=tenPreprocessedTwo, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

    #     tenFlow = torch.nn.functional.interpolate(input=netNetwork(tenPreprocessedOne, tenPreprocessedTwo), size=(intHeight, intWidth), mode='bilinear', align_corners=False)

    #     tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
    #     tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

    #     return tenFlow[0, :, :, :].cpu()
       
    # end
    
    # def estimate(self, tenOne, tenTwo):
    #     """
    #     Compute optical flow between two images.
    #     Expects tenOne and tenTwo as torch.Tensors of shape (3, H, W).
    #     Returns a flow field tensor of shape (2, H, W) on CPU.
    #     """
    #     # Check that both images have the same dimensions
    #     assert tenOne.shape[1] == tenTwo.shape[1], "Heights must match"
    #     assert tenOne.shape[2] == tenTwo.shape[2], "Widths must match"

    #     intWidth = tenOne.shape[2]
    #     intHeight = tenOne.shape[1]

    #     # For example, if using fixed dimensions 640x480:
    #     # assert intWidth == 640, f"Expected width 640, got {intWidth}"
    #     # assert intHeight == 480, f"Expected height 480, got {intHeight}"

    #     # Add batch dimension and move to CUDA
    #     tenPreprocessedOne = tenOne.cuda().view(1, 3, intHeight, intWidth)
    #     tenPreprocessedTwo = tenTwo.cuda().view(1, 3, intHeight, intWidth)

    #     # Compute the dimensions that are multiples of 64
    #     intPreprocessedWidth = int(math.ceil(intWidth / 64.0) * 64)
    #     intPreprocessedHeight = int(math.ceil(intHeight / 64.0) * 64)

    #     # Resize inputs to the required size
    #     tenPreprocessedOne = torch.nn.functional.interpolate(tenPreprocessedOne,
    #         size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
    #     tenPreprocessedTwo = torch.nn.functional.interpolate(tenPreprocessedTwo,
    #         size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

    #     # Perform forward pass (without tracking gradients)
    #     with torch.no_grad():
    #         tenFlow = self(tenPreprocessedOne, tenPreprocessedTwo)

    #     # Resize the flow back to the original size
    #     tenFlow = torch.nn.functional.interpolate(tenFlow,
    #         size=(intHeight, intWidth), mode='bilinear', align_corners=False)

    #     # Scale the flow values according to the resizing factors
    #     tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
    #     tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

    #     # Remove batch dimension and bring to CPU
    #     return tenFlow[0].cpu()

