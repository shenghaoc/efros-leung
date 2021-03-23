#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <algorithm>
using namespace cv;
using namespace std;

int scale = 3;
int threshold_value = 0;
int threshold_type = THRESH_BINARY;
int const max_binary_value = 1;

int sumNeighbors(Point p, Mat mask)
{
	int neighbors_x = p.x - 1;
	int neighbors_y = p.y - 1;
	int neighbors_width = 3;
	int neighbors_height = 3;
	// If either row or col is 0, position of top left is affected, need to adjust
	// Otherwise just shrink the rect

	if (p.x == 0)
	{
		neighbors_x = 0;
		neighbors_width--;
	}

	if (p.y == 0)
	{
		neighbors_y = 0;
		neighbors_height--;
	}

	if (p.x == mask.cols - 1)
	{
		neighbors_width--;
	}

	if (p.y == mask.rows - 1)
	{
		neighbors_height--;
	}

	return sum(mask(Rect(neighbors_x, neighbors_y, neighbors_width, neighbors_height)))[0];
}


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
	Mat GrayImage;
	Mat GraySampleImage;
	// Convert to grayscale for to avoid handling multiple channels
	cvtColor(Image, GrayImage, COLOR_RGB2GRAY, 0);
	cvtColor(SampleImage, GraySampleImage, COLOR_RGB2GRAY, 0);
	// Binary mask ensures dilation ignores regions within already filled pixels
	threshold(GrayImage, mask, threshold_value, max_binary_value, threshold_type);
	dilate(mask, UnfilledNeighbors, element);
	// Enlarged filled pixel region - filled pixel region = border formed by 
	// pixels along border of filled pixel region
	UnfilledNeighbors -= mask;
	vector<Point> PixelList; // Vector to be sorted by lambda expression
	findNonZero(UnfilledNeighbors, PixelList);
	random_shuffle(PixelList.begin(), PixelList.end()); // Randomly permute
	// Sort by decreasing number of filled neighbor pixels
	std::sort(PixelList.begin(), PixelList.end(), [mask](Point a, Point b) {
		return sumNeighbors(a, mask) > sumNeighbors(b, mask);
	});

	// Need to convert to floating point number to multiply with Gaussian kernel
	GrayImage.convertTo(GrayImage, CV_64F);
	Mat Template = GrayImage(Rect(0, 0, 5, 5)); // Grab a window
	Mat kernel_1d = getGaussianKernel(5, 1); // Get 1D Gaussian kernel
	// Convolution same as multiplication by transverse in this case
	Mat kernel_2d = kernel_1d * kernel_1d.t(); // Get 2D Gaussian kernel
	int TotWeight = sum(Template.mul(kernel_2d))[0];

	namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Display window", UnfilledNeighbors); // Show our image inside it.
	waitKey(0); // Wait for a keystroke in the window
	imwrite("result1.jpg", UnfilledNeighbors); // Save synthesized image
	return 0;
}