#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
using namespace cv;
using namespace std;
int main()
{
    Mat SampleImage;
    SampleImage = imread("texture1.jpg", IMREAD_COLOR); // Read the file
    if (SampleImage.empty()) // Check for invalid input
    {
        cout << "Could not open or find the sample image" << std::endl;
        return -1;
    }
    // Let synthesized image be 9 times the area of the sample image
    Mat Image = Mat::zeros(3 * SampleImage.rows, 3 * SampleImage.cols, CV_8UC3);
    // Copy sample image to upper left corner of synthesized image
    SampleImage.copyTo(Image(Rect(0, 0, SampleImage.cols, SampleImage.rows)));
    namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
    imshow("Display window", Image); // Show our image inside it.
    waitKey(0); // Wait for a keystroke in the window
    imwrite("result1.jpg", Image); // Save synthesized image
    return 0;
}