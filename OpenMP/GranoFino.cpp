#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <omp.h>

using namespace cv;
using namespace std;

#define _CRT_SECURE_NO_WARNINGS

#define IMAGEPATH "Imagenes\\lenna-32.jpg"
#define DISPLAY_RESULT_IMAGES true
#define REP_NUM 20


#ifdef WIN32
#define ourImread(filename, isColor) cvLoadImage(filename, isColor)
#else
#define ourImread(filename, isColor) imread(filename, isColor)
#endif

int main(int argc, char** argv)
{

	double time_b1 = INFINITY;
	
	Mat originalImage = ourImread(IMAGEPATH, CV_LOAD_IMAGE_COLOR);   // Read the file

	if (!originalImage.data)                              // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		waitKey(5000);
		return -1;
	}

	// Ex. b.1
	Mat new_image = Mat::zeros(originalImage.size(), originalImage.type());
	double alpha = 2.0, beta = 50; 

	for (int i = 0; i < REP_NUM; i++) {
		double time_spent, start_time, finish_time;

		start_time = omp_get_wtime();

		for (int y = 0; y < originalImage.rows; y++) {
			for (int x = 0; x < originalImage.cols; x++) {
				for (int c = 0; c < 3; c++) {
					new_image.at<Vec3b>(y, x)[c] =
						saturate_cast<uchar>(alpha*(originalImage.at<Vec3b>(y, x)[c]) + beta);
				}
			}
		}

		finish_time = omp_get_wtime();

		time_spent = finish_time - start_time;
		if (time_spent < time_b1)
			time_b1 = time_spent;
	}

	// Ex. b.2	
	Mat new_image2 = Mat::zeros(originalImage.size(), originalImage.type());
	double time_b2 = INFINITY;

	for (int j = 0; j < REP_NUM; j++) {
		double start_time, finish_time, time_spent;

		uchar *myData1 = originalImage.data;		uchar *myData2 = new_image2.data;		int stride = originalImage.step;
		start_time = omp_get_wtime();

		for (int y = 0; y < originalImage.rows; y++) {

			uchar *p1 = &(myData1[y * stride]);			uchar *p2 = &(myData2[y * stride]);			while (*p1) {								for (int color = 0; color < 3; color++) {					int value = alpha * (*p1) + beta;
					if (value > 255) {
						value = 255;
					}
					*p2 = value;

					p1++; 
					p2++;
				}
			}
		}
		finish_time = omp_get_wtime();

		time_spent = finish_time - start_time;

		if (time_spent < time_b2)
			time_b2 = time_spent;
	}



	if (DISPLAY_RESULT_IMAGES) {		

		cvNamedWindow("Image 1", CV_WINDOW_AUTOSIZE);// Create a window for display.
		cvShowImage("Image 1", cvCloneImage(&(IplImage) new_image));// Show our image inside it.	

		cvNamedWindow("Image 2", CV_WINDOW_AUTOSIZE);// Create a window for display.
		cvShowImage("Image 2", cvCloneImage(&(IplImage) new_image2));// Show our image inside it.

	}

	cout << "Methods mode execution:\n Min time spent: " << time_b1 << endl;
	cout << endl;
	cout << "Data mode execution:\n Min time spent: " << time_b2 << endl;
	cout << endl;

	
	waitKey(0);
	system("pause"); // this fragment avoids autoclosing the console if DISPLAY_RESULT_IMAGES is set to true

	return 0;
}
