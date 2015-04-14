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

void loadImages(char* filename, vector<Mat_<double> >& imgs) {
	FILE* fp = fopen(filename, "rb");

	int magic_number;
	fread(&magic_number, 4, 1, fp);

	int number_of_images;
	fread(&number_of_images, 4, 1, fp);

	int rows, cols;
	fread(&rows, 4, 1, fp);
	fread(&cols, 4, 1, fp);

	imgs.resize(number_of_images);

	for (int i = 0; i < number_of_images; ++i) {
		// 画像データを読み込む
		float* data = new float[rows * cols];
		fread(data, 4, rows * cols, fp);

		// Matオブジェクトにコピー
		imgs[i] = Mat_<uchar>(rows, cols);
		for (int r = 0; r < rows; ++r) {
			for (int c = 0; c < cols; ++c) {
				imgs[i](r, c) = data[r * cols + c];
			}
		}
	}
}

void sampleIMAGES(vector<Mat_<double> >& imgs, int num_patches, int patchsize, Mat_<double>& X) {
	X = Mat_<float>(patchsize * patchsize, num_patches);

	for (int i = 0; i < num_patches; ++i) {
		int img_id = rand() % imgs.size();

		int r0 = 0;
		if (imgs[img_id].rows > patchsize) {
			r0 = rand() % (imgs[img_id].rows - patchsize);
		}
		int c0 = 0;
		if (imgs[img_id].cols > patchsize) {
			c0 = rand() % (imgs[img_id].cols - patchsize);
		}

		/* //Matlabがcolumn majorでreshapeするので、とりあえず、コメントアウトしておく。
		for (int r = 0; r < patchsize; ++r) {
			for (int c = 0; c < patchsize; ++c) {
				X(r * patchsize + c, i) = imgs[img_id](r0 + r, c0 + c);
			}
		}
		*/
		for (int c = 0; c < patchsize; ++c) {
			for (int r = 0; r < patchsize; ++r) {
				X(c * patchsize + r, i) = imgs[img_id](r0 + r, c0 + c);
			}
		}

	}
	
	// 各列の平均値を計算
	Mat_<double> X_mean;
	reduce(X, X_mean, 0, CV_REDUCE_AVG);
	repeat(X_mean, X.rows, 1, X_mean);

	// 各列のの平均値を0にそろえる
	X -= X_mean;

	// 標準偏差を計算
	Scalar mean, stddev;
	meanStdDev(X, mean, stddev);

	// 各列の値が[-1, 1]の範囲になるようnormalizeする
	X = cv::max(cv::min(X, stddev.val[0] * 3.0), -stddev.val[0] * 3.0) / stddev.val[0] / 3.0;

	// [0.1,0.9]の範囲になるようnormalizeする
	X = (X + 1.0) * 0.4 + 0.1;
}

void test(int numpatches, int patchsize, int hiddenSize, int maxIter, double lambda, double beta, double sparsityParam, double learningRate) {
	// 以下の行は、最初に１回だけ、小さいデータセットを作成するために必要。
	//MNISTLoader::saveFirstNImages("train-images.idx3-ubyte", 10000, "images10000.idx3-ubyte");

	vector<Mat_<double> > imgs;
	//MNISTLoader::loadImages("images10000.idx3-ubyte", imgs, false);
	loadImages("images.dat", imgs);

	Mat_<double> X;
	sampleIMAGES(imgs, numpatches, patchsize, X);

	AutoEncoder ae(X, hiddenSize);
	for (int iter = 0; iter < maxIter; ++iter) {
		Updates updates = ae.train(lambda, beta, sparsityParam);
		ae.update(updates, learningRate);
		cout << iter << ": cost=" << updates.cost << endl;
	}
	ae.visualize("weights.png");
}

int main() {
	test(10000, // numpatches
		8,		// patchsize
		25,		// hiddenSize
		400,	// maxIter
		0.0001, // lambda
		3,		// beta
		0.01,	// sparsityParam
		0.4		// learningRate
		);

	return 0;
}
