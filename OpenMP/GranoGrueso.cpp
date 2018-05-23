#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <omp.h>

using namespace cv;
using namespace std;

#define _CRT_SECURE_NO_WARNINGS

#define IMAGE_NAME string("arnold-6MB")
#define IMAGE_EXTENSION string(".jpg")
#define IMAGES_DIR string("Imagenes\\")
#define DISPLAY_RESULT_IMAGES false
#define STORE_RESULT_IMAGES false
#define REP_NUM 1


#ifdef WIN32
#define ourImread(filename, isColor) cvLoadImage(filename, isColor)
#else
#define ourImread(filename, isColor) imread(filename, isColor)
#endif

void function1(Mat *img) {

	//difuminado

	GaussianBlur(*img, *img, Size(3, 3), 0, 0);

	//conversion color grises	

	cvtColor(*img, *img, COLOR_BGR2GRAY);

	//aplicar funcion thresholdç

	threshold(*img, *img, 150, 255, 1); // threshold_type = inverted

}

void function2(Mat *img) {
	//difuminado

	GaussianBlur(*img, *img, Size(3, 3), 0, 0);

	//conversion color grises

	cvtColor(*img, *img, COLOR_BGR2GRAY);

	//ecualización del histograma

	equalizeHist(*img, *img);

}

void function3(Mat *img) {
	Mat grad_x, grad_y;

	//difuminado

	GaussianBlur(*img, *img, Size(3, 3), 0, 0);

	//conversion color grises

	cvtColor(*img, *img, COLOR_BGR2GRAY);

	//aplicar el operador de Sobel	

	Sobel(*img, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(grad_x, grad_x);

	Sobel(*img, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(grad_y, grad_y);

	addWeighted(grad_x, 0.5, grad_y, 0.5, 0, *img);

}

void function4(Mat *img) {
	Mat grad_x, grad_y;

	//difuminado

	GaussianBlur(*img, *img, Size(3, 3), 0, 0);

	//conversion color grises

	cvtColor(*img, *img, COLOR_BGR2GRAY);

	//aplicar el operador de laplace

	Laplacian(*img, grad_x, CV_16S);
	convertScaleAbs(grad_x, grad_x);

	Laplacian(*img, grad_y, CV_16S);
	convertScaleAbs(grad_y, grad_y);

	addWeighted(grad_x, 0.5, grad_y, 0.5, 0, *img);

}


int main(int argc, char** argv)
{
	
	double parallelTime = INFINITY, seqTime = INFINITY;
	Mat imageArray[4];

	Mat originalImage = ourImread(IMAGES_DIR + IMAGE_NAME + IMAGE_EXTENSION, CV_LOAD_IMAGE_COLOR);   // Read the file

	if (!originalImage.data)                              // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		waitKey(5000);
		return -1;
	}

	

	for (int r = 0; r < REP_NUM; r++) {

		double startTime, finishTime, timeSpent;

		// Clone the image and insert into array 
		for (int i = 0; i < 4; i++) {
			imageArray[i] = originalImage.clone();
		}

		// Save the current time
		startTime = omp_get_wtime();

		// Execute filters
#pragma omp parallel sections
		{
#pragma omp section
			function1(&imageArray[0]);
#pragma omp section
			function2(&imageArray[1]);
#pragma omp section
			function3(&imageArray[2]);
#pragma omp section
			function4(&imageArray[3]);
		}
		// Save the current time
		finishTime = omp_get_wtime();

		timeSpent = finishTime - startTime;

		if (timeSpent < parallelTime)
			parallelTime = timeSpent;
	}

	for (int r = 0; r < REP_NUM; r++) {

		double startTime, finishTime, timeSpent;

		// Clone the image and insert into array 
		for (int i = 0; i < 4; i++) {
			imageArray[i] = originalImage.clone();
		}

		// Save the current time
		startTime = omp_get_wtime();

		// Execute filters
		function1(&imageArray[0]);
		function2(&imageArray[1]);
		function3(&imageArray[2]);
		function4(&imageArray[3]);
	
		// Save the current time
		finishTime = omp_get_wtime();

		timeSpent = finishTime - startTime;

		if (timeSpent < seqTime)
			seqTime = timeSpent;
	}


	if (DISPLAY_RESULT_IMAGES) {
		for (int i = 0; i < 4; i++) {
			string windowName = string("Image ") + to_string(i + 1);

			cvNamedWindow(windowName.c_str(), CV_WINDOW_AUTOSIZE);// Create a window for display.
			cvShowImage(windowName.c_str(), cvCloneImage(&(IplImage)imageArray[i]));// Show our image inside it.
		}
	}

	if (STORE_RESULT_IMAGES) {
		for (int i = 0; i < 4; i++) {
			imwrite(IMAGES_DIR + IMAGE_NAME + "-filterset"+to_string(i+1)+".jpg", imageArray[i]);			
		}
	}

	cout << "Parallel execution:\n Min time spent: " << parallelTime << endl;
	cout << endl;
	cout << "Sequential execution:\n Min time spent: " << seqTime << endl;
	cout << endl;

	waitKey(0);
	system("pause"); // this fragment avoids autoclosing the console if DISPLAY_RESULT_IMAGES is set to true

	return 0;
}
