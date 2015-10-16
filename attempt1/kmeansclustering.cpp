/*
// k-means clustering //
float pointsdata[200 * 2];

int cnt = 0;
for (int i = 0; i < number_of_features; i++)
{
pointsdata[cnt] = frame1_features[i].x;
cnt++;
pointsdata[cnt] = frame1_features[i].y;
cnt++;

}

int cluster_count = 10;

Mat points(number_of_features, 2, CV_32F, pointsdata);
Mat labels;
Mat centers(cluster_count, 1, points.type());

kmeans(points, cluster_count, labels, TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 100, 1.0), 5, KMEANS_PP_CENTERS, centers);

std::vector<float> neil;
neil.assign((float*)centers.datastart, (float*)centers.dataend);

float* array_sort_of = &neil[0];
for (size_t i = 0; i < cluster_count*2; i++)
{
cout << array_sort_of[i] << " ";
}

cout << "Center: \n" << centers << endl;
cout << "Labels: \n" << labels << endl;


CvPoint c1, c2, c3, c4, c5, c6, c7, c8, c9, c10;
c1.x = array_sort_of[0];
c1.y = array_sort_of[1];
c2.x = array_sort_of[2];
c2.y = array_sort_of[3];
c3.x = array_sort_of[4];
c3.y = array_sort_of[5];
c4.x = array_sort_of[6];
c4.y = array_sort_of[7];
c5.x = array_sort_of[8];
c5.y = array_sort_of[9];
c6.x = array_sort_of[10];
c6.y = array_sort_of[11];
c7.x = array_sort_of[12];
c7.y = array_sort_of[13];
c8.x = array_sort_of[14];
c8.y = array_sort_of[15];
c9.x = array_sort_of[16];
c9.y = array_sort_of[17];
c10.x = array_sort_of[18];
c10.y = array_sort_of[19];
cvCircle(frame1, c1, 50, CV_RGB(0, 0, 255), 1, CV_AA, 0);
cvCircle(frame1, c2, 50, CV_RGB(0, 0, 255), 1, CV_AA, 0);
cvCircle(frame1, c3, 50, CV_RGB(0, 0, 255), 1, CV_AA, 0);
cvCircle(frame1, c4, 50, CV_RGB(0, 0, 255), 1, CV_AA, 0);
cvCircle(frame1, c5, 50, CV_RGB(0, 0, 255), 1, CV_AA, 0);
cvCircle(frame1, c6, 50, CV_RGB(0, 0, 255), 1, CV_AA, 0);
cvCircle(frame1, c7, 50, CV_RGB(0, 0, 255), 1, CV_AA, 0);
cvCircle(frame1, c8, 50, CV_RGB(0, 0, 255), 1, CV_AA, 0);
cvCircle(frame1, c9, 50, CV_RGB(0, 0, 255), 1, CV_AA, 0);
cvCircle(frame1, c10, 50, CV_RGB(0, 0, 255), 1, CV_AA, 0);
// end of k-means clustering //
*/