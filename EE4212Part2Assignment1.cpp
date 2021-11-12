#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream> 
#include <sstream>  
#include <algorithm>
#include <random>
#include <regex>
#include <limits>

using namespace cv;
using namespace std;

int scale = 2;
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
		cerr << "Please enter an odd number as argument for the size of the neighborhood window!" << endl;
		return -1;
	}

	istringstream arg(argv[1]);
	int WindowSize;
	if (!(arg >> WindowSize))
	{
		cerr << "Invalid number: " << argv[1] << endl;
		return -1;
	}
	else if (!arg.eof())
	{
		cerr << "Trailing characters after number: " << argv[1] << endl;
		return -1;
	}

	if (WindowSize % 2 == 0)
	{
		cerr << "Not an odd number!" << endl;
		return -1;
	}

	Mat kernel_1d = getGaussianKernel(WindowSize, WindowSize / sigmaDiv, CV_32F); // Get 1D Gaussian kernel
	// Convolution same as multiplication by transverse in this case
	Mat kernel_2d = kernel_1d * kernel_1d.t(); // Get 2D Gaussian kernel

	string folder("texture*.jpg");
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
		// waitKey(0); // Wait for a keystroke in the window, uncomment to use
		imwrite(outputName, Image); // Save synthesized image
		destroyWindow(outputName);
	}
	return 0;
}

void GrowImage(Mat SampleImage, Mat& Image, int WindowSize, Mat GaussMask)
{
	// Let synthesized image be scale^2 times the area of the sample image
	// Copy sample image to upper left corner of synthesized image
	copyMakeBorder(SampleImage, Image, 0, (scale - 1) * SampleImage.rows, 0, (scale - 1) * SampleImage.cols, BORDER_CONSTANT);

	Mat mask = Mat::zeros(Image.size(), CV_32FC1);
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

			Mat Template;
			copyMakeBorder(GrayImage(Rect(Point(neighbors_x, neighbors_y), Size(neighbors_width, neighbors_height))), Template,
				padded_y, WindowSize - padded_y - neighbors_height,
				padded_x, WindowSize - padded_x - neighbors_width, BORDER_CONSTANT);

			Mat ValidMask;
			copyMakeBorder(mask(Rect(Point(neighbors_x, neighbors_y), Size(neighbors_width, neighbors_height))), ValidMask,
				padded_y, WindowSize - padded_y - neighbors_height,
				padded_x, WindowSize - padded_x - neighbors_width, BORDER_CONSTANT);

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
				mask.at<float>(PixelList[i]) = 1;
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
	Mat sqrtGaussMask;
	sqrt(GaussMask, sqrtGaussMask); // TM_SQDIFF squares everything including mask

	flip(GrayTemplate, GrayTemplate, -1);
	flip(ValidMask, ValidMask, -1);

	Mat SSD;

	// matchTemplate() starts from top left, need to pad and start at equivalent of
	// Point(-WindowSize / 2, -WindowSize / 2), end at equivalent of
	// Point(GraySampleImage.cols - 1 + WindowSize / 2, GraySampleImage.rows - 1 + WindowSize / 2)
	Mat paddedGraySampleImage;
	copyMakeBorder(GraySampleImage, paddedGraySampleImage,
		WindowSize / 2, WindowSize / 2, WindowSize / 2, WindowSize / 2,
		BORDER_CONSTANT);

	// Apparently the quadruple loop is similar to correlation (or convolution) and FFT can speed things up significantly
	// OpenCV happens to have implemented template matching with squared differences as an option
	// Hint from http://www.cs.umd.edu/~djacobs/CMSC733/PS2.pdf
	// Check out the links below for the math behind this
	// https://www.cs.umd.edu/~djacobs/CMSC426/Convolution.pdf
	// https://www.ics.uci.edu/~fowlkes/class/cs216/hwk2/hwk2.pdf
	matchTemplate(paddedGraySampleImage, GrayTemplate, SSD, TM_SQDIFF, ValidMask.mul(sqrtGaussMask) / TotWeight);

	// FFT or float data type may have rounding errors
	// Some 0s can be non-zero numbers with very large negative exponents
	// Need to search manually instead of minMaxLoc
	double min = numeric_limits<float>::max();
	for (int i = 0; i < GraySampleImage.cols; i++)
	{
		for (int j = 0; j < GraySampleImage.rows; j++)
		{
			// min search has to check and ignore negative numbers
			// Set to 0 to avoid repeated comparison
			SSD.at<float>(Point(i, j)) = SSD.at<float>(Point(i, j)) < 0 ? 0 : SSD.at<float>(Point(i, j));
			min = SSD.at<float>(Point(i, j)) < min ? SSD.at<float>(Point(i, j)) : min; // Check for min
		}
	}

	for (int i = 0; i < GraySampleImage.cols; i++)
	{
		for (int j = 0; j < GraySampleImage.rows; j++)
		{
			if (SSD.at<float>(Point(i, j)) <= min * (1 + ErrThreshold))
			{
				BestMatches.emplace_back(Point(i, j));
				errorList.emplace_back(SSD.at<float>(Point(i, j)));
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