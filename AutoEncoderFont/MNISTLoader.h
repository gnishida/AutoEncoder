#pragma once

/**
 * MNISTデータを読み込み、入力データ行列を作成する。
 */

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <vector>

using namespace std;
using namespace cv;

class MNISTLoader {
protected:
	MNISTLoader() {}

public:
	static void saveFirstNImages(char* inputFileName, int outputNum, char* outputFileName);
	static void loadImages(char* filename, vector<Mat_<double> >& imgs);

private:
	static int BE2LE(unsigned char* buff);
	static void LE2BE(int val, unsigned char* buff);
};

