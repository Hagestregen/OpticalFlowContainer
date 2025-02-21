#!/usr/bin/env python

import torch
from .correlation import correlation  # Import the custom correlation layer

# Global dictionary for backwarp grids
backwarp_tenGrid = {}

def backwarp(tenInput, tenFlow):
    """Warps an input tensor using a flow field."""
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, -1).repeat(1, 1, tenFlow.shape[2], 1)
        tenVer = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, -1, 1).repeat(1, 1, 1, tenFlow.shape[3])
        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([tenHor, tenVer], 1).cuda()
    
    tenFlow = torch.cat([
        tenFlow[:, 0:1, :, :] * (2.0 / (tenInput.shape[3] - 1.0)),
        tenFlow[:, 1:2, :, :] * (2.0 / (tenInput.shape[2] - 1.0))
    ], 1)
    
    return torch.nn.functional.grid_sample(
        input=tenInput,
        grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1),
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()

        class Features(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.netOne = torch.nn.Sequential(
                    torch.nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=3),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
                self.netTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
                self.netThr = torch.nn.Sequential(
                    torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
                self.netFou = torch.nn.Sequential(
                    torch.nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
                self.netFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
                self.netSix = torch.nn.Sequential(
                    torch.nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

            def forward(self, tenInput):
                tenOne = self.netOne(tenInput)
                tenTwo = self.netTwo(tenOne)
                tenThr = self.netThr(tenTwo)
                tenFou = self.netFou(tenThr)
                tenFiv = self.netFiv(tenFou)
                tenSix = self.netSix(tenFiv)
                return [tenOne, tenTwo, tenThr, tenFou, tenFiv, tenSix]

        class Matching(torch.nn.Module):
            def __init__(self, intLevel):
                super().__init__()
                self.fltBackwarp = [0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625][intLevel]
                if intLevel != 2:
                    self.netFeat = torch.nn.Sequential()
                elif intLevel == 2:
                    self.netFeat = torch.nn.Sequential(
                        torch.nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                    )
                if intLevel == 6:
                    self.netUpflow = None
                else:
                    self.netUpflow = torch.nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1, bias=False, groups=2)
                if intLevel >= 4:
                    self.netUpcorr = None
                else:
                    self.netUpcorr = torch.nn.ConvTranspose2d(49, 49, kernel_size=4, stride=2, padding=1, bias=False, groups=49)
                self.netMain = torch.nn.Sequential(
                    torch.nn.Conv2d(49, 128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(32, 2, kernel_size=[0, 0, 7, 5, 5, 3, 3][intLevel], stride=1, padding=[0, 0, 3, 2, 2, 1, 1][intLevel])
                )

            def forward(self, tenOne, tenTwo, tenFeaturesOne, tenFeaturesTwo, tenFlow):
                tenFeaturesOne = self.netFeat(tenFeaturesOne)
                tenFeaturesTwo = self.netFeat(tenFeaturesTwo)
                if tenFlow is not None:
                    tenFlow = self.netUpflow(tenFlow)
                if tenFlow is not None:
                    tenFeaturesTwo = backwarp(tenInput=tenFeaturesTwo, tenFlow=tenFlow * self.fltBackwarp)
                if self.netUpcorr is None:
                    tenCorrelation = torch.nn.functional.leaky_relu(
                        input=correlation.FunctionCorrelation(tenOne=tenFeaturesOne, tenTwo=tenFeaturesTwo, intStride=1),
                        negative_slope=0.1, inplace=False
                    )
                else:
                    tenCorrelation = self.netUpcorr(torch.nn.functional.leaky_relu(
                        input=correlation.FunctionCorrelation(tenOne=tenFeaturesOne, tenTwo=tenFeaturesTwo, intStride=2),
                        negative_slope=0.1, inplace=False
                    ))
                return (tenFlow if tenFlow is not None else 0.0) + self.netMain(tenCorrelation)

        class Subpixel(torch.nn.Module):
            def __init__(self, intLevel):
                super().__init__()
                self.fltBackward = [0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625][intLevel]
                if intLevel != 2:
                    self.netFeat = torch.nn.Sequential()
                elif intLevel == 2:
                    self.netFeat = torch.nn.Sequential(
                        torch.nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                    )
                self.netMain = torch.nn.Sequential(
                    torch.nn.Conv2d([0, 0, 130, 130, 194, 258, 386][intLevel], 128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(32, 2, kernel_size=[0, 0, 7, 5, 5, 3, 3][intLevel], stride=1, padding=[0, 0, 3, 2, 2, 1, 1][intLevel])
                )

            def forward(self, tenOne, tenTwo, tenFeaturesOne, tenFeaturesTwo, tenFlow):
                tenFeaturesOne = self.netFeat(tenFeaturesOne)
                tenFeaturesTwo = self.netFeat(tenFeaturesTwo)
                if tenFlow is not None:
                    tenFeaturesTwo = backwarp(tenInput=tenFeaturesTwo, tenFlow=tenFlow * self.fltBackward)
                return (tenFlow if tenFlow is not None else 0.0) + self.netMain(torch.cat([tenFeaturesOne, tenFeaturesTwo, tenFlow], 1))

        class Regularization(torch.nn.Module):
            def __init__(self, intLevel):
                super().__init__()
                self.fltBackward = [0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625][intLevel]
                self.intUnfold = [0, 0, 7, 5, 5, 3, 3][intLevel]
                if intLevel >= 5:
                    self.netFeat = torch.nn.Sequential()
                else:
                    self.netFeat = torch.nn.Sequential(
                        torch.nn.Conv2d([0, 0, 32, 64, 96, 128, 192][intLevel], 128, kernel_size=1, stride=1, padding=0),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                    )
                self.netMain = torch.nn.Sequential(
                    torch.nn.Conv2d([0, 0, 131, 131, 131, 131, 195][intLevel], 128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
                if intLevel >= 5:
                    self.netDist = torch.nn.Sequential(
                        torch.nn.Conv2d(32, [0, 0, 49, 25, 25, 9, 9][intLevel], kernel_size=[0, 0, 7, 5, 5, 3, 3][intLevel], stride=1, padding=[0, 0, 3, 2, 2, 1, 1][intLevel])
                    )
                else:
                    self.netDist = torch.nn.Sequential(
                        torch.nn.Conv2d(32, [0, 0, 49, 25, 25, 9, 9][intLevel], kernel_size=([0, 0, 7, 5, 5, 3, 3][intLevel], 1), stride=1, padding=([0, 0, 3, 2, 2, 1, 1][intLevel], 0)),
                        torch.nn.Conv2d([0, 0, 49, 25, 25, 9, 9][intLevel], [0, 0, 49, 25, 25, 9, 9][intLevel], kernel_size=(1, [0, 0, 7, 5, 5, 3, 3][intLevel]), stride=1, padding=(0, [0, 0, 3, 2, 2, 1, 1][intLevel]))
                    )
                self.netScaleX = torch.nn.Conv2d([0, 0, 49, 25, 25, 9, 9][intLevel], 1, kernel_size=1, stride=1, padding=0)
                self.netScaleY = torch.nn.Conv2d([0, 0, 49, 25, 25, 9, 9][intLevel], 1, kernel_size=1, stride=1, padding=0)

            def forward(self, tenOne, tenTwo, tenFeaturesOne, tenFeaturesTwo, tenFlow):
                tenDifference = (tenOne - backwarp(tenInput=tenTwo, tenFlow=tenFlow * self.fltBackward)).square().sum([1], True).sqrt().detach()
                tenDist = self.netDist(self.netMain(torch.cat([tenDifference, tenFlow - tenFlow.mean([2, 3], True), self.netFeat(tenFeaturesOne)], 1)))
                tenDist = tenDist.square().neg()
                tenDist = (tenDist - tenDist.max(1, True)[0]).exp()
                tenDivisor = tenDist.sum([1], True).reciprocal()
                tenScaleX = self.netScaleX(tenDist * torch.nn.functional.unfold(
                    input=tenFlow[:, 0:1, :, :], kernel_size=self.intUnfold, stride=1, padding=int((self.intUnfold - 1) / 2)
                ).view_as(tenDist)) * tenDivisor
                tenScaleY = self.netScaleY(tenDist * torch.nn.functional.unfold(
                    input=tenFlow[:, 1:2, :, :], kernel_size=self.intUnfold, stride=1, padding=int((self.intUnfold - 1) / 2)
                ).view_as(tenDist)) * tenDivisor
                return torch.cat([tenScaleX, tenScaleY], 1)

        self.netFeatures = Features()
        self.netMatching = torch.nn.ModuleList([Matching(intLevel) for intLevel in [2, 3, 4, 5, 6]])
        self.netSubpixel = torch.nn.ModuleList([Subpixel(intLevel) for intLevel in [2, 3, 4, 5, 6]])
        self.netRegularization = torch.nn.ModuleList([Regularization(intLevel) for intLevel in [2, 3, 4, 5, 6]])

    def forward(self, tenOne, tenTwo):
        """Computes optical flow between two images."""
        tenOne[:, 0, :, :] = tenOne[:, 0, :, :] - 0.411618
        tenOne[:, 1, :, :] = tenOne[:, 1, :, :] - 0.434631
        tenOne[:, 2, :, :] = tenOne[:, 2, :, :] - 0.454253
        tenTwo[:, 0, :, :] = tenTwo[:, 0, :, :] - 0.410782
        tenTwo[:, 1, :, :] = tenTwo[:, 1, :, :] - 0.433645
        tenTwo[:, 2, :, :] = tenTwo[:, 2, :, :] - 0.452793

        tenFeaturesOne = self.netFeatures(tenOne)
        tenFeaturesTwo = self.netFeatures(tenTwo)

        tenOne = [tenOne]
        tenTwo = [tenTwo]
        for intLevel in [1, 2, 3, 4, 5]:
            tenOne.append(torch.nn.functional.interpolate(
                input=tenOne[-1], size=(tenFeaturesOne[intLevel].shape[2], tenFeaturesOne[intLevel].shape[3]),
                mode='bilinear', align_corners=False
            ))
            tenTwo.append(torch.nn.functional.interpolate(
                input=tenTwo[-1], size=(tenFeaturesTwo[intLevel].shape[2], tenFeaturesTwo[intLevel].shape[3]),
                mode='bilinear', align_corners=False
            ))

        tenFlow = None
        for intLevel in [-1, -2, -3, -4, -5]:
            tenFlow = self.netMatching[intLevel](tenOne[intLevel], tenTwo[intLevel], tenFeaturesOne[intLevel], tenFeaturesTwo[intLevel], tenFlow)
            tenFlow = self.netSubpixel[intLevel](tenOne[intLevel], tenTwo[intLevel], tenFeaturesOne[intLevel], tenFeaturesTwo[intLevel], tenFlow)
            tenFlow = self.netRegularization[intLevel](tenOne[intLevel], tenTwo[intLevel], tenFeaturesOne[intLevel], tenFeaturesTwo[intLevel], tenFlow)

        return tenFlow * 20.0  # Scale factor as in original LiteFlowNet