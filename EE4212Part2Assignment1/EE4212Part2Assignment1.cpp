#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream> 
#include <sstream>  
#include <algorithm>
#include <random>
#include <regex>

using namespace cv;
using namespace std;

int scale = 2;
int threshold_value = 0;
int threshold_type = THRESH_BINARY;
int const max_binary_value = 1;
double ErrThreshold = 0.1;
double MaxErrThreshold = 0.3;
double sigmaDiv = 6.4;

random_device rd;   // non-deterministic generator
mt19937 gen(rd());  // to seed mersenne twister.

void GrowImage(Mat, Mat&, int, Mat);

void getNeighborParams(Point, Mat, int, int&, int&, int&, int&, int&, int&);

double sumNeighbors(Point, Mat, int);

void FindMatches(Mat, Mat, Mat, Mat, int, vector<Point>&, vector<double>&);

void RandomPick(vector<Point>, vector <double>, Point&, double&);

int main(int argc, char* argv[])
{
	if (argc != 2)
	{
		cout << "Please enter an odd number as argument for the size of the neighborhood window" << endl;
		return -1;
	}

	istringstream arg(argv[1]);
	int WindowSize;
	if (!(arg >> WindowSize)) 
	{
		cout << "Invalid number: " << argv[1] << endl;
	}
	else if (!arg.eof())
	{
		cout << "Trailing characters after number: " << argv[1] << endl;
	}

	Mat kernel_1d = getGaussianKernel(WindowSize, WindowSize / sigmaDiv); // Get 1D Gaussian kernel
	// Convolution same as multiplication by transverse in this case
	Mat kernel_2d = kernel_1d * kernel_1d.t(); // Get 2D Gaussian kernel

	string folder("texture1*.jpg");
	vector<String> fileNames;
	glob(folder, fileNames);

	for (int i = 0; i < fileNames.size(); i++)
	{
		Mat SampleImage = imread(fileNames[i], IMREAD_COLOR); // Read the file
		if (SampleImage.empty()) // Check for invalid input
		{
			cout << "Could not open or find the sample image" << endl;
			return -1;
		}

		Mat Image;
		GrowImage(SampleImage, Image, WindowSize, kernel_2d);

		String outputName = regex_replace(fileNames[i], regex("\\b(texture)([^ ]*)"), "synth$2");
		cout << "Finished synthesizing " << outputName << endl;
		namedWindow(outputName, WINDOW_AUTOSIZE); // Create a window for display.
		imshow(outputName, Image); // Show our image inside it.
		waitKey(0); // Wait for a keystroke in the window
		imwrite(outputName, Image); // Save synthesized image
		destroyWindow(outputName);
	}
	return 0;
}

void GrowImage(Mat SampleImage, Mat& Image, int WindowSize, Mat GaussMask)
{
	// Let synthesized image be scale^2 times the area of the sample image
	Image = Mat::zeros(scale * SampleImage.rows, scale * SampleImage.cols, CV_8UC3);
	// Copy sample image to upper left corner of synthesized image
	SampleImage.copyTo(Image(Rect(Point(0, 0), SampleImage.size())));

	Mat mask = Mat::zeros(Image.size(), CV_64FC1);
	mask(Rect(Point(0, 0), SampleImage.size())).setTo(Scalar(1));


	Mat UnfilledNeighbors;
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));

	Mat GrayImage;
	Image.convertTo(GrayImage, CV_32F, 1.0 / 255);
	cvtColor(GrayImage, GrayImage, COLOR_RGB2GRAY, 0);

	Mat GraySampleImage;
	SampleImage.convertTo(GraySampleImage, CV_32F, 1.0 / 255);
	cvtColor(GraySampleImage, GraySampleImage, COLOR_RGB2GRAY, 0);


	int numFilledPixels = SampleImage.cols * SampleImage.rows;
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
		sort(PixelList.begin(), PixelList.end(), [mask, WindowSize](Point a, Point b)
		{
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
			FindMatches(Template, GraySampleImage, ValidMask, GaussMask, WindowSize, BestMatches, errorList);
			double error = -1;
			Point BestMatch;
			RandomPick(BestMatches, errorList, BestMatch, error);
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
}

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
	int neighbors_x = -1;
	int neighbors_y = -1;
	int neighbors_width = -1;
	int neighbors_height = -1;
	int padded_x = -1;
	int padded_y = -1;
	getNeighborParams(p, mask, WindowSize, neighbors_x, neighbors_y, neighbors_width, neighbors_height, padded_x, padded_y);

	return sum(mask(Rect(neighbors_x, neighbors_y, neighbors_width, neighbors_height)))[0];
}

void FindMatches(Mat GrayTemplate, Mat GraySampleImage, Mat ValidMask, Mat GaussMask, int WindowSize, vector<Point>& BestMatches, vector<double>& errorList)
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
					if ((i - ii + WindowSize / 2 >= 0) && (i - ii + WindowSize / 2< GraySampleImage.cols)
						&& (j - jj + WindowSize /2>= 0) && (j - jj + WindowSize /2 < GraySampleImage.rows))
					{
						pixVal = GraySampleImage.at<float>(Point(i - ii + WindowSize/2, j - jj + WindowSize/2));
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
				BestMatches.emplace_back(Point(i, j));
				errorList.emplace_back(SSD.at<double>(Point(i, j)));
			}
		}
	}
}

void RandomPick(vector<Point> BestMatches, vector <double> errorList, Point& BestMatch, double& error)
{
	uniform_int_distribution<> dist(0, static_cast<int>(BestMatches.size()) - 1);
	int chosen = dist(gen);
	error = errorList[chosen];
	BestMatch = BestMatches[chosen];
}