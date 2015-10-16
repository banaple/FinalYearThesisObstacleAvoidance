/*
#include<opencv\cv.h>
#include<opencv\highgui.h>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

int framex = 854, framey = 480;

int main(int argc, char** argv)
{
	//string filename = "C:\\Users\\Neil Appleton\\Desktop\\Everything\\2015\\BEB 801\\f1topview.mp4";
	string filename = "C:\\Users\\Neil Appleton\\Desktop\\Everything\\2015\\BEB 801\\LARiverBikePath.mp4";
	VideoCapture capture(filename);
	Mat frame;

	if (!capture.isOpened())
		throw "Error when reading mp4";

	Point pt1total = (0, 0), pt2total = (0, 0), pt1totaltotal = (0, 0), pt2totaltotal = (0, 0);
	Point pt1totall = (0, 0), pt2totall = (0, 0), pt1totaltotall = (0, 0), pt2totaltotall = (0, 0);
	int counter = 1, counterr = 1;
	int houghVote = 0;
	float angleright = 119, angleleft = 31;


	for (;;)
	{
		capture >> frame;

		//canny edge
		Mat dst, cdst;
		Canny(frame, dst, 50, 200, 3);
		cvtColor(dst, cdst, CV_GRAY2BGR);
		//threshold(dst, cdst, 128, 255, THRESH_BINARY_INV);

		// detect lines
		vector<Vec2f> lines;
		// Hough Lines
		HoughLines(dst, lines, 1, CV_PI / 180, 115, 0, 0);


		// draw lines
		for (size_t i = 0; i < lines.size(); i++)
		{
			float rho = lines[i][0], theta = lines[i][1];
			// right side of lane
			if ((theta>CV_PI / 180 * 150 && theta<CV_PI / 180 * 180))
			{
				Point pt1, pt2;
				double a = cos(theta), b = sin(theta);
				double x0 = a*rho, y0 = b*rho;
				pt1.x = cvRound(x0 + 1000 * (-b));
				pt1.y = cvRound(y0 + 1000 * (a));
				pt2.x = cvRound(x0 - 1000 * (-b));
				pt2.y = cvRound(y0 - 1000 * (a));
				pt1total.x = pt1total.x + pt1.x;
				pt1total.y = pt1total.y + pt1.y;
				pt2total.x = pt2total.x + pt2.x;
				pt2total.y = pt2total.y + pt2.y;
				pt1totaltotal.x = pt1total.x / counter;
				pt1totaltotal.y = pt1total.y / counter;
				pt2totaltotal.x = pt2total.x / counter;
				pt2totaltotal.y = pt2total.y / counter;
				counter = counter + 1;
				line(cdst, pt1, pt2, Scalar(0, 0, 50), 3, CV_AA);
				cout << "pt1.y " << pt1.y << endl << "pt2.y " << pt2.y << endl << "pt1.x " << pt1.x << endl << "pt2.x " << pt2.x << endl << "angleright " << angleright << endl << endl;
			}
			angleright = 180 - atan2(pt2totaltotal.y - pt1totaltotal.y, pt2totaltotal.x - pt1totaltotal.x) * 180 / CV_PI;

			// left side of lane
			if ((theta<CV_PI / 180 * 60 && theta>CV_PI / 180 * 55))
			{
				Point pt1, pt2;
				double a = cos(theta), b = sin(theta);
				double x0 = a*rho, y0 = b*rho;
				pt1.x = cvRound(x0 + 1000 * (-b));
				pt1.y = cvRound(y0 + 1000 * (a));
				pt2.x = cvRound(x0 - 1000 * (-b));
				pt2.y = cvRound(y0 - 1000 * (a));
				pt1totall.x = pt1totall.x + pt1.x;
				pt1totall.y = pt1totall.y + pt1.y;
				pt2totall.x = pt2totall.x + pt2.x;
				pt2totall.y = pt2totall.y + pt2.y;
				pt1totaltotall.x = pt1totall.x / counterr;
				pt1totaltotall.y = pt1totall.y / counterr;
				pt2totaltotall.x = pt2totall.x / counterr;
				pt2totaltotall.y = pt2totall.y / counterr;
				counterr = counterr + 1;
				line(cdst, pt1, pt2, Scalar(0, 0, 50), 3, CV_AA);
				cout << "pt1.y " << pt1.y << endl << "pt2.y " << pt2.y << endl << "pt1.x " << pt1.x << endl << "pt2.x " << pt2.x << endl << "angleleft " << angleleft << endl << endl;
			}
			angleleft = 180 - atan2(pt1totaltotall.y - pt2totaltotall.y, pt1totaltotall.x - pt2totaltotall.x) * 180 / CV_PI;
		}

		cout << "angleleft " << angleleft << endl << "angleright " << angleright << endl << endl;

		// display average of left and right side of path
		line(cdst, pt1totaltotal, pt2totaltotal, Scalar(0, 0, 250), 3, CV_AA);
		line(cdst, pt1totaltotall, pt2totaltotall, Scalar(0, 0, 250), 3, CV_AA);

		Point totalpt1 = (0, 0), totalpt2 = (0, 0);
		totalpt1.x = (pt1totaltotal.x + pt1totaltotall.x) / 2;
		totalpt1.y = (pt1totaltotal.y + pt1totaltotall.y) / 2;
		totalpt2.x = (pt2totaltotal.x + pt2totaltotall.x) / 2;
		totalpt2.y = (pt2totaltotal.y + pt2totaltotall.y) / 2;
		cout << "pt1 " << totalpt1 << endl << "pt2 " << totalpt2 << endl << endl;
		line(cdst, totalpt1, totalpt2, Scalar(0, 0, 250), 3, CV_AA);

		// find intersection of 2 averaged lines
		int A1, B1, C1, A2, B2, C2;
		A1 = pt2totaltotal.y - pt1totaltotal.y;
		B1 = pt1totaltotal.x - pt2totaltotal.x;
		C1 = A1*pt1totaltotal.x + B1*pt1totaltotal.y;

		A2 = pt2totaltotall.y - pt1totaltotall.y;
		B2 = pt1totaltotall.x - pt2totaltotall.x;
		C2 = A2*pt1totaltotall.x + B2*pt1totaltotall.y;

		double det = A1*B2 - A2*B1;
		double interceptx=0, intercepty=0;
		if (det == 0)
		{
			cout << "lines are parallel" << endl;
		}
		else
		{
			interceptx = (B2*C1 - B1*C2) / det;
			intercepty = (A1*C2 - A2*C1) / det;
			cout << "x= " << interceptx << " y= " << intercepty << endl;
		}

		// draw line from origin to intersection
		Point intercept;
		intercept.x = interceptx;
		intercept.y = intercepty;
		circle(cdst, intercept, 4, Scalar(20, 20, 20), -1, 8);
		Point origin;
		origin.x = 420;
		origin.y = 460;
		circle(frame, origin, 4, Scalar(0, 0, 200), -1, 8);
		line(frame, intercept, origin, Scalar(0, 200, 0), 3, CV_AA);
		line(frame, intercept, origin, Scalar(0, 200, 0), 3, CV_AA);
		 
		


		namedWindow("Bike Path", 1);			//create a window with the name "Bike Path"
		namedWindow("Canny Edge Bike Path", 1); //create a window with the name "Canny Edge Detection Bike Path"
		imshow("Bike Path", frame);
		imshow("Canny Edge Bike Path", cdst);	//display the image which is stored in the 'cdst' in the "Canny Edge Detection Bike Path" window
		waitKey();							    // waits to display frame
	}
	//waitKey();
	
}

*/