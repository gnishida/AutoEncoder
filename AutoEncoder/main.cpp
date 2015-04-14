#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "MNISTLoader.h"
#include "AutoEncoder.h"

using namespace std;
using namespace cv;

/**
 * 関数x1^2 + 3*x1*x2の勾配を返却する。
 */
double simpleQuadraticFunction(const Mat_<double>& x) {
	return x(0, 0) * x(0, 0) + x(0, 0) * x(0, 1) * 3.0;
}

void sampleIMAGES(vector<Mat_<uchar> >& imgs, int num_patches, int patch_rows, int patch_cols, Mat_<double>& X) {
	X = Mat_<float>(patch_rows * patch_cols, num_patches);

	for (int i = 0; i < num_patches; ++i) {
		int img_id = rand() % imgs.size();
		int r0 = rand() % (imgs[img_id].rows - patch_rows);
		int c0 = rand() % (imgs[img_id].cols - patch_cols);

		for (int r = 0; r < patch_rows; ++r) {
			for (int c = 0; c < patch_cols; ++c) {
				X(r * patch_cols + c, i) = imgs[img_id](r0 + r, c0 + c);
			}
		}
	}

	Scalar mean, stddev;
	meanStdDev(X, mean, stddev);

	// [-1, 1]の範囲になるようnormalizeする
	X -= mean;
	X = cv::max(cv::min(X, stddev.val[0] * 3.0), -stddev.val[0] * 3.0) / stddev.val[0] / 3.0;

	// [0.1,0.9]の範囲になるようnormalizeする
	X = (X + 1.0) * 0.4 + 0.1;
}

int main() {
	// 以下の行は、最初に１回だけ、小さいデータセットを作成するために必要。
	//MNISTLoader::saveFirstNImages("train-images.idx3-ubyte", 10000, "images10000.idx3-ubyte");

	vector<Mat_<uchar> > imgs;
	MNISTLoader::loadImages("images10000.idx3-ubyte", imgs, true);

	Mat_<double> X;
	//sampleIMAGES(imgs, 10000, 8, 8, X);
	sampleIMAGES(imgs, 100, 8, 8, X);

	AutoEncoder ae(X, 25);
	for (int iter = 0; iter < 500; ++iter) {
		Updates updates = ae.train(0.0001, 3);
		ae.update(updates, 0.1);
	}
	ae.visualize("weights.png");

	return 0;
}
