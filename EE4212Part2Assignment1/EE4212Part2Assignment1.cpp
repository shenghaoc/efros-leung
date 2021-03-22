#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int scale = 3;
int threshold_value = 0;
int threshold_type = THRESH_BINARY;
int const max_binary_value = 255;

int main()
{
	Mat SampleImage;
	SampleImage = imread("texture1.jpg", IMREAD_COLOR); // Read the file
	if (SampleImage.empty()) // Check for invalid input
	{
		cout << "Could not open or find the sample image" << std::endl;
		return -1;
	}

	// Let synthesized image be scale^2 times the area of the sample image
	Mat Image = Mat::zeros(scale * SampleImage.rows, scale * SampleImage.cols, CV_8UC3);
	// Copy sample image to upper left corner of synthesized image
	SampleImage.copyTo(Image(Rect(0, 0, SampleImage.cols, SampleImage.rows)));

	Mat UnfilledNeighbors;
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	Mat mask;
	// Binary mask ensures dilation ignores regions within already filled pixels
	threshold(Image, mask, threshold_value, max_binary_value, threshold_type);
	dilate(mask, UnfilledNeighbors, element);
	// Enlarged filled pixel region - filled pixel region = border formed by 
	// pixels along border of filled pixel region
	UnfilledNeighbors -= mask;
	// Convert to grayscale for findNonZero() to work
	cvtColor(UnfilledNeighbors, UnfilledNeighbors, COLOR_RGB2GRAY, 0);
	Mat PixelList;
	findNonZero(UnfilledNeighbors, PixelList);

	namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Display window", UnfilledNeighbors); // Show our image inside it.
	waitKey(0); // Wait for a keystroke in the window
	imwrite("result1.jpg", UnfilledNeighbors); // Save synthesized image
	return 0;
}