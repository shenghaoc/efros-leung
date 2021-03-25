#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <algorithm>
#include <random>
using namespace cv;
using namespace std;

int scale = 2;
int threshold_value = 0;
int threshold_type = THRESH_BINARY;
int const max_binary_value = 1;
double ErrThreshold = 0.1;
double MaxErrThreshold = 0.3;

random_device rd;   // non-deterministic generator
mt19937 gen(rd());  // to seed mersenne twister.

void getNeighborParams(Point p, Mat Image, int WindowSize, int& neighbors_x, int& neighbors_y, int& neighbors_width, int& neighbors_height, int& padded_x, int& padded_y)
{
	neighbors_x = p.x - WindowSize / 2;
	neighbors_y = p.y - WindowSize / 2;
	neighbors_width = WindowSize;
	neighbors_height = WindowSize;
	padded_x = 0;
	padded_y = 0;
	if (neighbors_x < 0)
	{
		padded_x = -neighbors_x;
		neighbors_width -= padded_x;
		neighbors_x = 0;
	}

	if (neighbors_y < 0)
	{
		padded_y = -neighbors_y;
		neighbors_height -= padded_y;
		neighbors_y = 0;
	}

	if (neighbors_x > Image.cols - WindowSize - 1)
	{
		neighbors_width -= neighbors_x - (Image.cols - WindowSize - 1);
	}

	if (neighbors_y > Image.rows - WindowSize - 1)
	{
		neighbors_height -= neighbors_y - (Image.rows - WindowSize - 1);
	}
}

double sumNeighbors(Point p, Mat mask, int WindowSize)
{
	int neighbors_x = 0;
	int neighbors_y = 0;
	int neighbors_width = 3;
	int neighbors_height = 3;
	int padded_x = 0;
	int padded_y = 0;
	getNeighborParams(p, mask, WindowSize, neighbors_x, neighbors_y, neighbors_width, neighbors_height, padded_x, padded_y);

	return sum(mask(Rect(neighbors_x, neighbors_y, neighbors_width, neighbors_height)))[0];
}

void FindMatches(Mat GrayTemplate, Mat GraySampleImage, Mat ValidMask, Mat GaussMask, vector<Point>& BestMatches, vector<double>& errorList)
{
	double TotWeight = sum(ValidMask.mul(GaussMask))[0];

	Mat SSD = Mat::zeros(GraySampleImage.rows, GraySampleImage.cols, CV_64FC1);
	double dist = 0;
	double pixVal = -1;
	for (int i = 0; i < GraySampleImage.cols; i++)
	{
		for (int j = 0; j < GraySampleImage.rows; j++)
		{
			for (int ii = 0; ii < GrayTemplate.cols; ii++)
			{
				for (int jj = 0; jj < GrayTemplate.rows; jj++)
				{
					if ((i - ii + GrayTemplate.cols / 2 >= 0) && (i - ii + GrayTemplate.cols / 2 < GraySampleImage.cols) && (j - jj + GrayTemplate.rows / 2 >= 0) && (j - jj + GrayTemplate.rows / 2 < GraySampleImage.rows))
					{
						pixVal = GraySampleImage.at<float>(Point(i - ii + GrayTemplate.cols / 2, j - jj + GrayTemplate.rows / 2));
					}
					else
					{
						pixVal = 0;
					}
					dist = GrayTemplate.at<float>(Point(ii, jj)) - pixVal;
					dist *= dist;
					SSD.at<double>(Point(i, j)) += dist * ValidMask.at<double>(Point(ii, jj)) * GaussMask.at<double>(Point(ii, jj));


				}
			}
				SSD.at<double>(Point(i, j)) /= TotWeight;
		}
	}

	double min = -1;
	double max = -1;
	minMaxLoc(SSD, &min, &max);
	for (int i = 0; i < GraySampleImage.cols; i++)
	{
		for (int j = 0; j < GraySampleImage.rows; j++)
		{
			if (SSD.at<double>(Point(i, j)) <= min * (1 + ErrThreshold))
			{
				BestMatches.push_back(Point(i, j));
				errorList.push_back(SSD.at<double>(Point(i, j)));
			}
		}
	}
}

Point RandomPick(vector<Point> BestMatches, vector <double> errorList, double& error)
{
	uniform_int_distribution<> dist(0, static_cast<int>(BestMatches.size()) - 1);
	int chosen = dist(gen);
	error = errorList[chosen];
	return BestMatches[chosen];
}


int main()
{
	int WindowSize = 5;
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
	SampleImage.copyTo(Image(Rect(Point(0, 0), SampleImage.size())));

	Mat mask = Mat::zeros(Image.size(), CV_64FC1);
	mask(Rect(Point(0, 0), SampleImage.size())).setTo(Scalar(1));

	int numFilledPixels = SampleImage.cols * SampleImage.rows;

	Mat UnfilledNeighbors;
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));


	Mat GrayImage;
	Image.convertTo(GrayImage, CV_32F, 1.0 / 255);
	cvtColor(GrayImage, GrayImage, COLOR_RGB2GRAY, 0);

	Mat GraySampleImage;
	SampleImage.convertTo(GraySampleImage, CV_32F, 1.0 / 255);
	cvtColor(GraySampleImage, GraySampleImage, COLOR_RGB2GRAY, 0);

	Mat kernel_1d = getGaussianKernel(WindowSize, WindowSize / 6.4); // Get 1D Gaussian kernel
	// Convolution same as multiplication by transverse in this case
	Mat kernel_2d = kernel_1d * kernel_1d.t(); // Get 2D Gaussian kernel


	int progress = 0;
	int neighbors_x = -1;
	int neighbors_y = -1;
	int neighbors_width = -1;
	int neighbors_height = -1;
	int padded_x = -1;
	int padded_y = -1;
	while (numFilledPixels < scale * scale * SampleImage.cols * SampleImage.rows)
	{
		cout << "Progress: " << numFilledPixels << endl;
		progress = 0;

		dilate(mask, UnfilledNeighbors, element);
		UnfilledNeighbors -= mask;

		vector<Point> PixelList; // Vector to be sorted by lambda expression
		findNonZero(UnfilledNeighbors, PixelList);
		shuffle(PixelList.begin(), PixelList.end(), gen); // Randomly permute
		// Sort by decreasing number of filled neighbor pixels
		std::sort(PixelList.begin(), PixelList.end(), [mask, WindowSize](Point a, Point b) {
			return sumNeighbors(a, mask, WindowSize) > sumNeighbors(b, mask, WindowSize);
		});

		for (int i = 0; i < PixelList.size(); i++)
		{
			getNeighborParams(PixelList[i], mask, WindowSize, neighbors_x, neighbors_y, neighbors_width, neighbors_height, padded_x, padded_y);

			Mat Template = Mat::zeros(Size(WindowSize, WindowSize), CV_32FC1);
			GrayImage(Rect(Point(neighbors_x, neighbors_y), Size(neighbors_width, neighbors_height)))
				.copyTo(Template(Rect(Point(padded_x, padded_y), Size(neighbors_width, neighbors_height)))); // Grab a window

			Mat ValidMask = Mat::zeros(Size(WindowSize, WindowSize), CV_64FC1);
			mask(Rect(Point(neighbors_x, neighbors_y), Size(neighbors_width, neighbors_height)))
				.copyTo(ValidMask(Rect(Point(padded_x, padded_y), Size(neighbors_width, neighbors_height))));

			vector <double> errorList;
			vector<Point> BestMatches;
			FindMatches(Template, GraySampleImage, ValidMask, kernel_2d, BestMatches, errorList);
			double error = -1;
			Point BestMatch = RandomPick(BestMatches, errorList, error);
			if (error < MaxErrThreshold)
			{
				SampleImage(Rect(BestMatch, Size(1, 1))).copyTo(Image(Rect(PixelList[i], Size(1, 1))));
				GraySampleImage(Rect(BestMatch, Size(1, 1))).copyTo(GrayImage(Rect(PixelList[i], Size(1, 1))));
				mask.at<double>(PixelList[i]) = 1;
				progress = 1;
				numFilledPixels++;
			}
		}
		if (progress == 0)
		{
			MaxErrThreshold *= 1.1;
		}
	}

	namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Display window", Image); // Show our image inside it.
	waitKey(0); // Wait for a keystroke in the window
	imwrite("result1.jpg", Image); // Save synthesized image
	return 0;
}