
#include <iostream>
#include <sstream>
#include <string.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <cmath>
#include <vector>

using namespace std;
using namespace cv;

#include<opencv\cv.h>
#include<opencv\highgui.h>

static const double pi = 3.14159265358979323846;

inline static double square(int a)
{
	return a * a;
}



// This is just an inline that allocates images.  I did this to reduce clutter in the
// actual computer vision algorithmic code.  Basically it allocates the requested image
// unless that image is already non-NULL.  It always leaves a non-NULL image as-is even
// if that image's size, depth, and/or channels are different than the request.
inline static void allocateOnDemand(IplImage **img, CvSize size, int depth, int channels)
{
	if (*img != NULL)	return;

	*img = cvCreateImage(size, depth, channels);
	if (*img == NULL)
	{
		fprintf(stderr, "Error: Couldn't allocate image.  Out of memory?\n");
		exit(-1);
	}
}



// DB SCAN //
vector<int> regionQuery(vector<KeyPoint> *keypoints, KeyPoint *keypoint, float eps)
{
	float dist;
	vector<int> retKeys;
	for (int i = 0; i< keypoints->size(); i++)
	{
		dist = sqrt(pow((keypoint->pt.x - keypoints->at(i).pt.x), 2) + pow((keypoint->pt.y - keypoints->at(i).pt.y), 2));
		if (dist <= eps && dist != 0.0f)
		{
			retKeys.push_back(i);
		}
	}
	return retKeys;
}

vector<vector<KeyPoint>> DBSCAN_keypoints(vector<KeyPoint> *keypoints, float eps, int minPts)
{
	vector<vector<KeyPoint>> clusters;
	vector<bool> clustered;
	vector<int> noise;
	vector<bool> visited;
	vector<int> neighborPts;
	vector<int> neighborPts_;
	int c;

	int noKeys = keypoints->size();

	//init clustered and visited
	for (int k = 0; k < noKeys; k++)
	{
		clustered.push_back(false);
		visited.push_back(false);
	}

	//C =0;
	c = 0;
	clusters.push_back(vector<KeyPoint>()); //will stay empty?

	//for each unvisted point P in dataset keypoints
	for (int i = 0; i < noKeys; i++)
	{
		if (!visited[i])
		{
			//Mark P as visited
			visited[i] = true;
			neighborPts = regionQuery(keypoints, &keypoints->at(i), eps);
			if (neighborPts.size() < minPts)
				//Mark P as Noise
				noise.push_back(i);
			else
			{
				clusters.push_back(vector<KeyPoint>());
				c++;
				//expand cluster
				// add P to cluster c
				clusters[c].push_back(keypoints->at(i));
				clustered[i] = true;
				//for each point P' in neighborPts
				for (int j = 0; j < neighborPts.size(); j++)
				{
					//if P' is not visited
					if (!visited[neighborPts[j]])
					{
						//Mark P' as visited
						visited[neighborPts[j]] = true;
						neighborPts_ = regionQuery(keypoints, &keypoints->at(neighborPts[j]), eps);
						if (neighborPts_.size() >= minPts)
						{
							neighborPts.insert(neighborPts.end(), neighborPts_.begin(), neighborPts_.end());
						}
					}
					// if P' is not yet a member of any cluster
					// add P' to cluster c
					if (!clustered[neighborPts[j]])
						clusters[c].push_back(keypoints->at(neighborPts[j]));
					clustered[neighborPts[j]] = true;
				}
			}

		}
	}
	return clusters;
}

// DB SCAN END //


// Spline or De Boors Algorithm
Point deBoor(int k, int degree, int i, double x, double* knots, Point *ctrlPoints)
{   // Please see wikipedia page for detail
	// note that the algorithm here kind of traverses in reverse order
	// comapred to that in the wikipedia page
	if (k == 0)
		return ctrlPoints[i];	
	else
	{
		double alpha = (x - knots[i]) / (knots[i + degree + 1 - k] - knots[i]);
		return (deBoor(k - 1, degree, i - 1, x, knots, ctrlPoints)*(1 - alpha) + deBoor(k - 1, degree, i, x, knots, ctrlPoints)*alpha);
	}
}

// Spline end



int main(void)
{

	// Create an object that decodes the input video stream.
	CvCapture *input_video = cvCaptureFromFile("C:\\Users\\Neil Appleton\\Desktop\\Everything\\2015\\BEB 801\\bike2compressed.mp4");



	//CvCapture *input_video = cvCaptureFromFile("C:\\Users\\Neil Appleton\\Desktop\\Everything\\2015\\BEB 801\\bike1compressed.mp4");
	//CvCapture *input_video = cvCaptureFromFile("C:\\Users\\Neil Appleton\\Desktop\\Everything\\2015\\BEB 801\\LARiverBikePath.mp4");
	if (input_video == NULL)
	{
		// Either the video didn't exist OR it uses a codec OpenCV doesn't support.
		fprintf(stderr, "Error: Can't open video.\n");
		return -1;
	}

	// Read the video's frame size out of the AVI.
	CvSize frame_size;
	frame_size.height =
		(int)cvGetCaptureProperty(input_video, CV_CAP_PROP_FRAME_HEIGHT);
	frame_size.width =
		(int)cvGetCaptureProperty(input_video, CV_CAP_PROP_FRAME_WIDTH);

	// Determine the number of frames in the AVI.
	long number_of_frames;
	// Go to the end of the AVI (ie: the fraction is "1") 
	cvSetCaptureProperty(input_video, CV_CAP_PROP_POS_AVI_RATIO, 1.);
	// Now that we're at the end, read the AVI position in frames
	number_of_frames = (int)cvGetCaptureProperty(input_video, CV_CAP_PROP_POS_FRAMES);
	// Return to the beginning
	cvSetCaptureProperty(input_video, CV_CAP_PROP_POS_FRAMES, 0.);

	// Create a windows called "Optical Flow" for visualizing the output.
	// Have the window automatically change its size to match the output.
	cvNamedWindow("Optical Flow", CV_WINDOW_AUTOSIZE);

	long current_frame = 0;
	while (true)
	{
		static IplImage *frame = NULL, *frame1 = NULL, *frame1_1C = NULL, *frame2_1C = NULL, *eig_image = NULL, *temp_image = NULL, *pyramid1 = NULL, *pyramid2 = NULL;

		// Go to the frame we want.  Important if multiple frames are queried in
		// the loop which they of course are for optical flow.  Note that the very
		// first call to this is actually not needed. (Because the correct position
		// is set outsite the for() loop.)
		cvSetCaptureProperty(input_video, CV_CAP_PROP_POS_FRAMES, current_frame);

		// Get the next frame of the video.
		// IMPORTANT!  cvQueryFrame() always returns a pointer to the _same_
		// memory location.  So successive calls:
		// frame1 = cvQueryFrame();
		// frame2 = cvQueryFrame();
		// frame3 = cvQueryFrame();
		// will result in (frame1 == frame2 && frame2 == frame3) being true.
		// The solution is to make a copy of the cvQueryFrame() output.
		frame = cvQueryFrame(input_video);
		if (frame == NULL)
		{
			// Why did we get a NULL frame?  We shouldn't be at the end
			fprintf(stderr, "Error: Hmm. The end came sooner than we thought.\n");
			return -1;
		}
		// Allocate another image if not already allocated.
		// Image has ONE channel of color (ie: monochrome) with 8-bit "color" depth.
		// This is the image format OpenCV algorithms actually operate on (mostly).
		allocateOnDemand(&frame1_1C, frame_size, IPL_DEPTH_8U, 1);
		// Convert whatever the AVI image format is into OpenCV's preferred format.
		// AND flip the image vertically.  Flip is a shameless hack.  OpenCV reads
		// in AVIs upside-down by default.  (No comment :-))
		cvConvertImage(frame, frame1_1C);

		// We'll make a full color backup of this frame so that we can draw on it.
		// (It's not the best idea to draw on the static memory space of cvQueryFrame().)
		allocateOnDemand(&frame1, frame_size, IPL_DEPTH_8U, 3);
		cvConvertImage(frame, frame1);

		// Get the second frame of video.  Same principles as the first
		frame = cvQueryFrame(input_video);
		if (frame == NULL)
		{
			fprintf(stderr, "Error: Hmm. The end came sooner than we thought.\n");
			return -1;
		}
		allocateOnDemand(&frame2_1C, frame_size, IPL_DEPTH_8U, 1);
		cvConvertImage(frame, frame2_1C);

		// Shi and Tomasi Feature Tracking!

		// Preparation: Allocate the necessary storage
		allocateOnDemand(&eig_image, frame_size, IPL_DEPTH_32F, 1);
		allocateOnDemand(&temp_image, frame_size, IPL_DEPTH_32F, 1);

		// Preparation: This array will contain the features found in frame 1
		const int featnum = 100;
		CvPoint2D32f frame1_features[featnum];

		// Preparation: BEFORE the function call this variable is the array size
		// (or the maximum number of features to find).  AFTER the function call
		// this variable is the number of features actually found
		int number_of_features;

		// I'm hardcoding this at 400.  But you should make this a #define so that you can
		// change the number of features you use for an accuracy/speed tradeoff analysis.
		//
		
		number_of_features = featnum;

		// Actually run the Shi and Tomasi algorithm!!
		// "frame1_1C" is the input image.
		// "eig_image" and "temp_image" are just workspace for the algorithm.
		// The first ".01" specifies the minimum quality of the features (based on the eigenvalues).
		// The second ".01" specifies the minimum Euclidean distance between features.
		// "NULL" means use the entire input image.  You could point to a part of the image.
		// WHEN THE ALGORITHM RETURNS:
		// "frame1_features" will contain the feature points.
		// "number_of_features" will be set to a value <= 400 indicating the number of feature points found
		cvGoodFeaturesToTrack(frame1_1C, eig_image, temp_image, frame1_features, &number_of_features, .01, .01, NULL, 3, 0.4);

		// Pyramidal Lucas Kanade Optical Flow!

		// This array will contain the locations of the points from frame 1 in frame 2
		CvPoint2D32f frame2_features[featnum];

		// The i-th element of this array will be non-zero if and only if the i-th feature of
		// frame 1 was found in frame 2.
		char optical_flow_found_feature[featnum];

		// The i-th element of this array is the error in the optical flow for the i-th feature
		// of frame1 as found in frame 2.  If the i-th feature was not found (see the array above)
		// I think the i-th entry in this array is undefined
		float optical_flow_feature_error[featnum];

		// This is the window size to use to avoid the aperture problem (see slide "Optical Flow: Overview")
		CvSize optical_flow_window = cvSize(3, 3);

		// This termination criteria tells the algorithm to stop when it has either done 20 iterations or when
		// epsilon is better than .3.  You can play with these parameters for speed vs. accuracy but these values
		// work pretty well in many situations
		CvTermCriteria optical_flow_termination_criteria
			= cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, .3);

		// This is some workspace for the algorithm.
		// (The algorithm actually carves the image into pyramids of different resolutions.)
		allocateOnDemand(&pyramid1, frame_size, IPL_DEPTH_8U, 1);
		allocateOnDemand(&pyramid2, frame_size, IPL_DEPTH_8U, 1);

		// Actually run Pyramidal Lucas Kanade Optical Flow!!
		// "frame1_1C" is the first frame with the known features.
		// "frame2_1C" is the second frame where we want to find the first frame's features.
		// "pyramid1" and "pyramid2" are workspace for the algorithm.
		// "frame1_features" are the features from the first frame.
		// "frame2_features" is the (outputted) locations of those features in the second frame.
		// "number_of_features" is the number of features in the frame1_features array.
		// "optical_flow_window" is the size of the window to use to avoid the aperture problem.
		// "5" is the maximum number of pyramids to use.  0 would be just one level.
		// "optical_flow_found_feature" is as described above (non-zero iff feature found by the flow).
		// "optical_flow_feature_error" is as described above (error in the flow for this feature).
		// "optical_flow_termination_criteria" is as described above (how long the algorithm should look).
		// "0" means disable enhancements.  (For example, the second array isn't pre-initialized with guesses.)
		cvCalcOpticalFlowPyrLK(frame1_1C, frame2_1C, pyramid1, pyramid2, frame1_features, frame2_features, number_of_features, optical_flow_window, 5, optical_flow_found_feature, optical_flow_feature_error, optical_flow_termination_criteria, CV_LKFLOW_PYR_A_READY | CV_LKFLOW_PYR_B_READY);

		// For fun (and debugging :)), let's draw the flow field.
		for (int i = 0; i < number_of_features; i++)
		{
			// If Pyramidal Lucas Kanade didn't really find the feature, skip it
			if (optical_flow_found_feature[i] == 0)	continue;

			int line_thickness;				line_thickness = 1;
			// CV_RGB(red, green, blue) is the red, green, and blue components
			// of the color you want, each out of 255.
			CvScalar line_color;			line_color = CV_RGB(255, 0, 0);

			// Let's make the flow field look nice with arrows

			// The arrows will be a bit too short for a nice visualization because of the high framerate
			// (ie: there's not much motion between the frames).  So let's lengthen them by a factor of 3.
			CvPoint p, q;
			p.x = (int)frame1_features[i].x;
			p.y = (int)frame1_features[i].y;
			q.x = (int)frame2_features[i].x;
			q.y = (int)frame2_features[i].y;

			double angle;		angle = atan2((double)p.y - q.y, (double)p.x - q.x);
			double hypotenuse;	hypotenuse = sqrt(square(p.y - q.y) + square(p.x - q.x));

			//samples[i][1] = double(q.y);


			// Here we lengthen the arrow by a factor of three.
			//q.x = (int)(p.x - 3 * hypotenuse * cos(angle));
			//q.y = (int)(p.y - 3 * hypotenuse * sin(angle));

			// Now we draw the main line of the arrow
			// "frame1" is the frame to draw on.
			// "p" is the point where the line begins.
			// "q" is the point where the line stops.
			// "CV_AA" means antialiased drawing.
			// "0" means no fractional bits in the center cooridinate or radius
			cvLine(frame1, p, q, line_color, line_thickness, CV_AA, 0);
			// Now draw the tips of the arrow.  I do some scaling so that the
			// tips look proportional to the main line of the arrow.
			p.x = (int)(q.x + 9 * cos(angle + pi / 4));
			p.y = (int)(q.y + 9 * sin(angle + pi / 4));
			cvLine(frame1, p, q, line_color, line_thickness, CV_AA, 0);
			p.x = (int)(q.x + 9 * cos(angle - pi / 4));
			p.y = (int)(q.y + 9 * sin(angle - pi / 4));
			cvLine(frame1, p, q, line_color, line_thickness, CV_AA, 0);

		}


		// Start using DBSCAN
		float pointsdata[featnum * 2];

		int cnt = 0;
		for (int i = 0; i < number_of_features; i++)
		{
			pointsdata[cnt] = (int)frame1_features[i].x;	
			cnt++;
			pointsdata[cnt] = (int)frame1_features[i].y;
			cnt++;
		}

		vector<KeyPoint> keypointFeatures;

		int cntt = 0;
		for (int i = 0; i < number_of_features; i++)
		{
			keypointFeatures.push_back(KeyPoint(pointsdata[cntt], pointsdata[cntt+1], 0));
			cntt = cntt + 2;
		}
		
		vector<KeyPoint>* ptrFeatures;
		ptrFeatures = &keypointFeatures;

		// DBSCAN_keypoints(keypoints, eps, minPts)
		vector<vector<KeyPoint>> my_clusters;
		my_clusters = DBSCAN_keypoints(ptrFeatures, 10, 5);


		if (my_clusters.size() > 0)
		{
			for (int ii = 0; ii < my_clusters.size(); ii++)
			{
				double max_distx = 0; double min_distx = 10000; double max_disty = 0; double min_disty = 10000;
				for (int zz = 0; zz < my_clusters[ii].size(); zz++)
				{
					if (my_clusters[ii][zz].pt.x < min_distx) min_distx = my_clusters[ii][zz].pt.x;
					if (my_clusters[ii][zz].pt.x > max_distx) max_distx = my_clusters[ii][zz].pt.x;
					if (my_clusters[ii][zz].pt.y < min_disty) min_disty = my_clusters[ii][zz].pt.y;
					if (my_clusters[ii][zz].pt.y > max_disty) max_disty = my_clusters[ii][zz].pt.y;
				}
				double midx; double midy;
				midx = (min_distx + max_distx) / 2;
				midy = (min_disty + max_disty) / 2;
				// filter out top of image and a little of the sides
				// why did i use this ratio = cause of vertex point
				if ((midy > ((frame_size.height) * (1 - (8.5 / 10.5))) && ((midx > 345) && (midx < 615))))
				{
					cvRectangle(frame1, Point(min_distx, min_disty), Point(max_distx, max_disty), CV_RGB(0, 0, 255), 1);
					cout << frame_size.width << " = width" << endl; // width = 960
					cout << frame_size.height << " = height" << endl; // height = 540
					cout << midx << " = midx" << endl;
					cout << midy << " = midy" << endl << endl;

				}

				Point deborrr;
				double knots[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
				Point happy = ((frame_size.width) / 2, frame_size.height);
				//happy[0] = { ((frame_size.width) / 2, frame_size.height) };
				//happy[1] = { (640, 2750), ((frame_size.width) / 2, 0) };
				Point *ctrlctrl;
				ctrlctrl = &happy;

				deborrr = deBoor(3, 3, 2, 2.5, knots, ctrlctrl);
				
				printf("%d\n", deborrr.x);
				cout << int (deborrr.x) << " = deborrr" << endl << endl;
				cout << int (deborrr.y) << " = deborrr" << endl << endl;
				//cvRectangle(frame1, Point(min_distx, min_disty), Point(max_distx, max_disty), CV_RGB(0, 0, 255), 1);
			}
		}
		

		

		// Now display the image we drew on.  Recall that "Optical Flow" is the name of
		// the window we created above.
		cvConvertImage(frame1, frame1);
		cvShowImage("Optical Flow", frame1);
		// And wait for the user to press a key (so the user has time to look at the image).
		// If the argument is 0 then it waits forever otherwise it waits that number of milliseconds.
		// The return value is the key the user pressed.
		int key_pressed;
		key_pressed = cvWaitKey();

		// If the users pushes "b" or "B" go back one frame.
		// Otherwise go forward one frame.
		if (key_pressed == 'b' || key_pressed == 'B')	current_frame--;
		else											current_frame++;
		// Don't run past the front/end of the AVI
		if (current_frame < 0)						current_frame = 0;
		if (current_frame >= number_of_frames - 1)	current_frame = number_of_frames - 2;
	}
}


