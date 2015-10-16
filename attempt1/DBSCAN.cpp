/*DBSCAN - density-based spatial clustering of applications with noise */
/*
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
*/