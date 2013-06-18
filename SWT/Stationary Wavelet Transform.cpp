#include "SWT2D.hpp"
#include <omp.h>   



 
void conv2(const Mat &img, const Mat& kernel, ConvolutionType type, Mat& dest, int flipcode) {
  Mat source = img;
  if(CONVOLUTION_FULL == type)
  {
    source = Mat();
    const int additionalRows = kernel.rows-1, additionalCols = kernel.cols-1;
    copyMakeBorder(img, source, (additionalRows+1)/2, additionalRows/2, (additionalCols+1)/2, additionalCols/2, BORDER_CONSTANT, Scalar(0));
  }
 
  Point anchor(kernel.cols - kernel.cols/2 - 1, kernel.rows - kernel.rows/2 - 1);
  int borderMode = BORDER_CONSTANT;
  Mat kernel_temp;
  flip(kernel,kernel_temp, flipcode);
  filter2D(source, dest, img.depth(), kernel_temp, anchor, 0, borderMode);
 




  if(CONVOLUTION_VALID == type) {
    dest = dest.colRange((kernel.cols-1)/2, dest.cols - kernel.cols/2)
               .rowRange((kernel.rows-1)/2, dest.rows - kernel.rows/2);
  }
}



void ExtendPeriod(const Mat &B, Mat &C, int level, Dimension _D)
{
	/* Check for correct values of levels */
	if (_D == Two)
	{
		
			int Level_Mat[3] = {2,4,8};               /* This tells how much the source Matrix must be expanded at edges */
			int inc_Per = Level_Mat[level];			  /* Calculate the expansion at that particular level */
			C = Mat::zeros(B.rows + inc_Per , B.cols + inc_Per , CV_32F);	 /* Create Matrix with expanded edges */
		

			/* Copy original Matrix into new Matrix */

			B.rowRange(0,B.rows).copyTo(C.rowRange((int)(inc_Per/2), B.rows + inc_Per/2).colRange((int)(inc_Per/2), B.cols + inc_Per/2));

			/* Copy the columns from original matrix to new Matrix for edge periodization */

		
			B.rowRange(0, B.rows).colRange(B.cols - inc_Per/2, B.cols).copyTo(C.rowRange(inc_Per/2, B.rows + inc_Per/2).colRange(0, inc_Per/2));
				
			B.rowRange(0, B.rows).colRange(0, inc_Per/2).copyTo(C.rowRange(inc_Per/2, B.rows + inc_Per/2).colRange(B.cols + inc_Per/2, C.cols));
		
		
			/* Copy the rows from original matrix to new Matrix for edge periodization */
		
			C.rowRange(B.rows , B.rows + inc_Per/2).copyTo(C.rowRange(0, inc_Per/2));
		
			C.rowRange(inc_Per/2, inc_Per).copyTo(C.rowRange(B.rows + inc_Per/2, C.rows));
		
	}
	else if (_D == One)
	{
			C = Mat::zeros(B.rows , B.cols + level , CV_32F);	 /* Create Matrix with expanded edges */

			/* Copy original Matrix into new Matrix */

			B.rowRange(0,B.rows).copyTo(C.colRange((int)(level/2), B.cols + level/2));

			/* Copy the columns from original matrix to new Matrix for edge periodization */

		
			B.rowRange(0, B.rows).colRange(B.cols - level/2, B.cols).copyTo(C.colRange(0, level/2));
				
			B.rowRange(0, B.rows).colRange(0, level/2).copyTo(C.colRange(B.cols + level/2, C.cols));
	}
}


void FilterBank(Mat &Kernel_High, Mat &Kernel_Low, MotherWavelet Type, FilterType Filter)
{
	if(Type == Haar)
	{
		/* Decide between Re-construction or Decomposition Filter bank */
		/* D is for decomposition, and R is for re-construction*/



		if (Filter == D)
		{
			/* Initiliaze Haar's Filter Bank */
	
			Kernel_High.at<float>(0,0) = (float) (-0.7071);
			Kernel_High.at<float>(0,1) = (float) (0.7071);
			Kernel_Low.at<float>(0,0) = (float) (0.7071);
			Kernel_Low.at<float>(0,1) = (float) (0.7071);
		}
		else if (Filter == R)
		{
			Kernel_High.at<float>(0,0) = (float) (0.7071);
			Kernel_High.at<float>(0,1) = (float) (-0.7071);
			Kernel_Low.at<float>(0,0) = (float) (0.7071);
			Kernel_Low.at<float>(0,1) = (float) (0.7071);
		}
	}
}

void KeepLoc(Mat &src, int Extension, int OriginalSize, Dimension _D)
{
		  /* Get rid of the Edges, and get an image out which is of original dimensions */
		  /* Note: This currently works for only Square matrices. Update needed. */
		if (_D == Two)
		{
		  int End = Extension + OriginalSize - 1;
		  Mat dst = Mat::zeros(OriginalSize, OriginalSize, CV_32F);
		  src.rowRange(Extension - 1, End).colRange(Extension - 1, End).copyTo(dst);
		  src = dst;
		}
		else if (_D == One)
		{
			int End = Extension + OriginalSize - 1;
			Mat dst = Mat::zeros(1, OriginalSize, CV_32F);
            src.colRange(Extension - 1, End).copyTo(dst);
		    src = dst;
		}
}


Mat DyadicUpsample(Mat &kernel, Dimension dim)
{
	if (dim == One)
	{
			/* Create a new Matrix with two rows */

			Mat temp = Mat::zeros(kernel.rows + 1, kernel.cols, CV_32F);
	
			/* Copy Kernel into first row */

			kernel.row(0).colRange(0, kernel.cols).copyTo(temp.row(0));
	
			/* Now traverse the Matrix in Zig-Zag manner, and insert the column  */
			/* values into a new column major Matrix */
			/* Note: There must be a faster way to do this in OpenCV. I will be updating it. */

			Mat Ret = Mat::zeros(kernel.cols * 2, 1, CV_32F);
			int Index = 0;
			for (int i = 0; i < (kernel.cols * 2)/2 ; i++)
			{
				temp.rowRange(0, temp.rows).col(i).copyTo(Ret.col(0).rowRange(Index, Index + 2));
				Index = Index + 2;
			}

			/* Take transpose converting column major matrix to row major */

			transpose(Ret, Ret);
			return Ret;
	}
	else if (dim == Two)
	{
			/* Create a new Matrix with double the number of kernel rows */

			Mat temp = Mat::zeros(kernel.rows * 2, kernel.cols * 2, CV_32F);
			int Index = 0, _Col = 0, _Row = 0;

			/* Update odd indexes of new row with kernel rows */

			for (int row = 0; row < kernel.rows * 2; row = row + 2)
			{
				_Col = 0;
				for (int col = 0; col < kernel.cols * 2; col = col + 2)
				{
					kernel.row(_Row).col(_Col).copyTo(temp.row(row).col(col));
					_Col = _Col + 1;
				}
				_Row = _Row + 1;
			}

			Mat temp_1 = Mat::zeros(temp.rows * 2, temp.cols, CV_32F);
			return temp;
	}
}


void SWT2(const Mat &src_Original, Mat &ca, Mat &ch, Mat &cd, Mat &cv, int Level, MotherWavelet Type)
{

	/* Right now only Haar is implemented */


	if (Type == Haar)
	{
		
		/* Decalre and Intialize helper Matrices */

		Mat Kernel_High = Mat::zeros(1, 2, CV_32F);
		Mat Kernel_Low = Mat::zeros(1, 2, CV_32F);
		Mat swa, swh, swv, swd;
		Mat src = src_Original;


		/* Initiliaze Filter Banks for Haar Transform */


		FilterBank(Kernel_High, Kernel_Low, Haar, D);


		/* The main loop for calculating Stationary 2D Wavelet Transform */


		for (int i = 0; i < Level; i++)
		{
		
			/* A temporary src Matrix for calculations */			

			Mat Extended_src = Mat::zeros(1,1, CV_32F);
			

			/* Extend source Matrix to deal with edge related issues */

			ExtendPeriod(src, Extended_src, i, Two);
			
			

			/* Helper Matrices */

			Mat y = Mat::zeros(1,1, CV_32F);
			Mat y1 = Mat::zeros(1,1, CV_32F);
			Mat z = Mat::zeros(1,1, CV_32F);
			
			/* Calculating Approximation coeffcients */

			conv2(Extended_src, Kernel_Low, CONVOLUTION_FULL, y, 0);
			
			transpose(y , y1);
			
			conv2(y1, Kernel_Low, CONVOLUTION_FULL, z, 0);
			
			transpose(z , swa);
						
			KeepLoc(swa, Kernel_Low.cols + 1, src.cols, Two);
			
			
			/* Calculating Horizontal coeffcients */
			
			

			conv2(y1, Kernel_High, CONVOLUTION_FULL, z, 0);
			
			transpose(z , swh);
			
			KeepLoc(swh, Kernel_Low.cols + 1, src.cols, Two);
			
			
			/* Calculating Vertical coeffcients */

			
			conv2(Extended_src, Kernel_High, CONVOLUTION_FULL, y, 0);

			transpose(y , y1);
			
			conv2(y1, Kernel_Low, CONVOLUTION_FULL, z, 0);
			
			transpose(z , swv);
			
			KeepLoc(swv, Kernel_Low.cols + 1, src.cols, Two);
			
			

			/* Calculating Diagonal coeffcients */


			conv2(y1, Kernel_High, CONVOLUTION_FULL, z, 0);

			transpose(z , swd);

			KeepLoc(swd, Kernel_Low.cols + 1, src.cols, Two);
			
			

			/* Upsamle Low and High Pass Filters */

			Kernel_High = DyadicUpsample(Kernel_High, One);
			Kernel_Low = DyadicUpsample(Kernel_Low, One);

			/* Create a vector of Matrices to store two channels */

			vector <Mat> temp;

			/* Split Hor, Ver, Diag and App into respective channels, copy the latest */
			/* calculated co-efficients into the channels, and merge them */

			split(ca, temp); swa.copyTo(temp[i]); merge(temp, ca);    /* Approximation coeffcients */

			split(ch, temp); swh.copyTo(temp[i]); merge(temp, ch);	  /* Horizontal coefficients */

			split(cv, temp); swv.copyTo(temp[i]); merge(temp, cv);    /* Vertical coeffcients */
			
			split(cd, temp); swd.copyTo(temp[i]); merge(temp, cd);    /* Diagonal coeffcients */
			

			/* Copy the Approximation co-efficients into this stage's source Mat */
			/* for next stage decomposition */

			swa.copyTo(src);

		}
	}
}


void SubMat(Mat &src, Mat &dst, SubMatType Type)
{
	int _Row = 0, _Col = 0;
	dst = Mat::zeros(src.rows/2, src.cols/2, CV_32F);
	if (Type == Odd)
	{
		for(int i = 1; i < src.rows; i = i + 2)
		{
			_Col = 0;
			for(int j = 1; j < src.cols; j = j + 2)
			{
				src.row(i).col(j).copyTo(dst.row(_Row).col(_Col));
				_Col = _Col + 1;
			}
			_Row = _Row + 1;
		}
	}	
	else if (Type == Even)
	{
		for(int i = 0; i < src.rows; i = i + 2)
		{
			_Col = 0;
			for(int j = 0; j < src.rows; j = j + 2)
			{
				src.row(i).col(j).copyTo(dst.row(_Row).col(_Col));
				_Col = _Col + 1;
			}
			_Row = _Row + 1;
		}
	}
}

Mat PartitionedWaveletTransform(Mat &src, Mat &kernel_1, Mat &kernel_2, int Level, int size)
{
		/* Helper matrices */

		Mat C = src, temp_1, temp_2;
		
		/* Dyadic upsample of src matrix */
		
		C = DyadicUpsample(C, Two);

		/* Extending period of matrix */

		ExtendPeriod(C, temp_1, Level, Two);	
		
		/* convolution to calculate Wavelet Transform */

		transpose(C,C);	
		conv2(C, kernel_1, CONVOLUTION_FULL, temp_2, 0);
		transpose(temp_2, temp_2);
				
		conv2(temp_2, kernel_2, CONVOLUTION_FULL, temp_2, 0);	
		KeepLoc(temp_2, kernel_1.rows + 1, size, Two);


		return temp_2;
}

void ISWT2(const Mat &dst, Mat &ca, Mat &ch, Mat &cd, Mat &cv, int Level, MotherWavelet Type)
{
		if (Type == Haar)
		{
			Mat Kernel_High = Mat::zeros(1, 2, CV_32F);
			Mat Kernel_Low = Mat::zeros(1, 2, CV_32F);
		
			/* Initiliaze Filter Banks for Haar Transform */

			FilterBank(Kernel_High, Kernel_Low, Haar, R);
			
			for (int i = 0;  i < Level; i++)
			{
				Mat C, H, D, V, X1, X2, temp_1, temp_2;


				/* Even Indexes  and Wavelet Transform for reconstruction*/

				SubMat(ca, C, Even);	
				SubMat(cd, D, Even);	
				SubMat(cv, V, Even);	
				SubMat(ch, H, Even);
				
				X1 = PartitionedWaveletTransform(C, Kernel_Low, Kernel_Low, i, ca.rows)
					+ PartitionedWaveletTransform(H, Kernel_High, Kernel_Low, i, ca.rows) 
					+ PartitionedWaveletTransform(V, Kernel_Low, Kernel_High, i, ca.rows)
					+ PartitionedWaveletTransform(D, Kernel_High, Kernel_High, i, ca.rows);

				

				/* Odd Indexes  and Wavelet Transform for re-construction*/

				SubMat(ca, C, Odd);	SubMat(cd, D, Odd);	SubMat(cv, V, Odd);	SubMat(ch, H, Odd);
				
				X2 = PartitionedWaveletTransform(C, Kernel_Low, Kernel_Low, i, ca.rows)
					+ PartitionedWaveletTransform(H, Kernel_High, Kernel_Low, i, ca.rows) 
					+ PartitionedWaveletTransform(V, Kernel_Low, Kernel_High, i, ca.rows)
					+ PartitionedWaveletTransform(D, Kernel_High, Kernel_High, i, ca.rows);

				
				
				temp_1 = Mat::zeros(X2.rows, X2.cols, CV_32F);
			

				/* Shifting the Even Indexed Wavelet Transformed matrix */

				X2.rowRange(0, X2.rows - 1).copyTo(temp_1.rowRange(1, X2.rows));
				X2.row(X2.rows - 1).copyTo(temp_1.row(0));
				temp_1.copyTo(X2);

				
				X2.colRange(0, X2.cols - 1).copyTo(temp_1.colRange(1, X2.cols));
				X2.col(X2.cols - 1).copyTo(temp_1.col(0));
				X2 = temp_1;

				/* Adding up and dividing by two the two yielded matrices*/
				

				ca = (X1 + X2) * 0.5;
			}

		}
}


void SWT(const Mat &src_Original, Mat &swa, Mat &swd, int Level, MotherWavelet Type)
{

	/* Right now only Haar is implemented */


	if (Type == Haar)
	{
		
		/* Decalre and Intialize helper Matrices */

		Mat Kernel_High = Mat::zeros(1, 2, CV_32F);
		Mat Kernel_Low = Mat::zeros(1, 2, CV_32F);
		swa = Mat::zeros(Level, src_Original.cols, CV_32F);
		swd = Mat::zeros(Level, src_Original.cols, CV_32F);;
		Mat src = src_Original;
		Mat y_Low, y_High;

		/* Initiliaze Filter Banks for Haar Transform */


		FilterBank(Kernel_High, Kernel_Low, Haar, D);

		

		/* The main loop for calculating Stationary 2D Wavelet Transform */


		for (int i = 0; i < Level; i++)
		{
		
			/* A temporary src Matrix for calculations */			

			Mat Extended_src;
			

			/* Extend source Matrix to deal with edge related issues */

			
			ExtendPeriod(src, Extended_src, Kernel_High.cols, One);
			
			
			/* Calculating Approximation coeffcients */

			
			conv2(Extended_src, Kernel_Low, CONVOLUTION_FULL, y_Low, 1);
			
			KeepLoc(y_Low, Kernel_Low.cols + 1, src.cols, One);

			


			
			
			y_Low.copyTo(swa.row(i));

			
			/* Calculating Detail coeffcients */
			
			
			conv2(Extended_src, Kernel_High, CONVOLUTION_FULL, y_High, 1);
			

			KeepLoc(y_High, Kernel_Low.cols + 1, src.cols, One);
			
			
			/*if (i == 2)
			{
				for (int row = 0; row < Extended_src.rows; row++)
				{
					for(int col = 0;  col < 10; col++)
					{	
						cout<<y_High.row(row).col(col)<<" ";
					}
					cout<<endl;
					break;
			   }
			}*/
			
			
			y_High.copyTo(swd.row(i));
			
			/* Upsamle Low and High Pass Filters */

			Kernel_High = DyadicUpsample(Kernel_High, One);
			Kernel_Low = DyadicUpsample(Kernel_Low, One);

			

			
			/* Copy the Approximation co-efficients into this stage's source Mat */
			/* for next stage decomposition */

			swa.row(i).copyTo(src);

		}
	}
}



void Test_conv()
{

		Mat Kernel_High = Mat::zeros(1, 2, CV_32F);
		Mat Kernel_Low = Mat::zeros(1, 2, CV_32F);

        FilterBank(Kernel_High, Kernel_Low, Haar, D);
		Mat a = Mat::zeros(1, 4, CV_32F);

		Kernel_High = DyadicUpsample(Kernel_High, One);
		Kernel_Low = DyadicUpsample(Kernel_Low, One);

		a.row(0).col(0) = 1;
		a.row(0).col(1) = 2;
		a.row(0).col(2) = 3;
		a.row(0).col(3) = 4;

		Mat y_High, y_Low;

		 
		conv2(a, Kernel_High, CONVOLUTION_FULL, y_High, 1);
		conv2(a, Kernel_Low, CONVOLUTION_FULL, y_Low, 1);
		/*for (int i = 0; i <  y_Low.rows; i++)
		{
			for(int j = 0; j <  y_Low.cols; j++)
			{	
				cout<< y_Low.row(i).col(j)<<" ";
			}
			cout<<endl;
			break;
		}	*/

}



int main ()
{
  /// Declare variables
  Mat src;

  char* window_name = "filter2D Demo";
  int level = 1;

  /// Load an image
  src = imread("J:/Work/Filter/Data/lena.jpg", CV_LOAD_IMAGE_GRAYSCALE);

  Test_conv();
  Mat swa = Mat::zeros(512,512,CV_32F);
  Mat swh = Mat::zeros(512,512,CV_32F);
  Mat swv = Mat::zeros(512,512,CV_32F);
  Mat swd = Mat::zeros(512,512,CV_32F);

  Mat a = Mat::zeros(1, 4, CV_32F);

  a.row(0).col(0) = 1;
  a.row(0).col(1) = 2;
  a.row(0).col(2) = 3;
  a.row(0).col(3) = 4;

 
  CvMLData b;
  b.read_csv("J:\\Work\\SWT_Denoise\\noisbloc.csv");
  
  Mat c;
  
  c = b.get_values();

  

  Mat swa_1, swd_1;

  SWT(c, swa_1, swd_1, 3, Haar);

  for (int i = 1; i < swa_1.rows; i++)
  {
	for(int j = 0; j < 10; j++)
	{	
		cout<<swd_1.row(i).col(j)<<" ";
	}
	cout<<endl;
	break;
  }



  if( !src.data )
  { return -1; }
   


  vector <Mat> Approx;

  /* Calling Stationary Wavelet Transform Function */
  double start = omp_get_wtime();
  SWT2(src, swa, swh, swd, swv, 1, Haar);
				
  ISWT2(src, swa, swh, swd, swv, 1, Haar);
  double end = omp_get_wtime() - start;
  return 0;
}
