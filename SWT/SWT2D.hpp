#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"

#include <iostream>
using namespace cv;
using namespace std;

#ifndef SWT2D_H
#define SWT2D_H

enum MotherWavelet 
{
  /* Haar Wavelet */
  Haar,
  
  /* Dmeyer */
  
  Dmey,
  
  /* Symmlets */
  Symm
};

/* I am terribly thankful for Timm Linder for modifying OpenCV's Filter2D function for Full 2d Convolution */

enum ConvolutionType {   /* Return the full convolution, including border */
  CONVOLUTION_FULL,
  /* Return only the part that corresponds to the original image */
  CONVOLUTION_SAME,
  /* Return only the submatrix containing elements that were not influenced by the border */
  CONVOLUTION_VALID
};

enum SubMatType {  
	/* Select the Odd rows and columns */
	Odd,
	/* Select the Even rows and columns*/
	Even
};

enum Dimension {  
	/* 1-D Signal */
	One,
	/* 2-D Signal */
	Two
};


enum FilterType {  
	/* Reconstruction */
	R,
	/* Decomposition */
	D
};


void ExtendPeriod(const Mat &B, Mat &C, int level, Dimension _D); 
void conv2(const Mat &img, const Mat& kernel, ConvolutionType type, Mat& dest, int flipcode) ;

void FilterBank(Mat &Kernel_High, Mat &Kernel_Low, MotherWavelet Type, FilterType Filter);
void KeepLoc(Mat &src, int Extension, int OriginalSize, Dimension _D);
void SWT2(const Mat &src_Original, Mat &ca, Mat &ch, Mat &cd, Mat &cv, int Level, MotherWavelet Type);
void ISWT2(const Mat &dst, Mat &ca, Mat &ch, Mat &cd, Mat &cv, int Level, MotherWavelet Type);
void SWT(const Mat &src_Original, Mat &swa, Mat &swd, int Level, MotherWavelet Type);
void ISWT(const Mat &src_Original, Mat &swa, Mat &swd, int Level, MotherWavelet Type);
void SubMat(Mat &src, Mat &dst, SubMatType Type);
Mat PartitionedWaveletTransform(Mat &src, Mat &kernel_1, Mat &kernel_2, int Level, int size);
Mat DyadicUpsample(Mat &kernel, Dimension dim);
void Test_conv();

#endif