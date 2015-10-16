#include<opencv\cv.h>
#include<opencv\highgui.h>

using namespace cv;

int main(){

	Mat image;

	VideoCapture cap;
	cap.open(0);

	namedWindow("window", 1);

	while (1){
		cap >> image;
		imshow("window", image);

	}
}