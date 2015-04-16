#include "stdafx.h"
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "MNISTLoader.h"
#include "AutoEncoder.h"
#include "optimization.h"

using namespace std;
using namespace cv;
using namespace alglib;

AutoEncoder* ae;

double lambda = 0.0001;
double beta = 3;
double sparsityParam = 0.01;

void function1_grad(const real_1d_array &x, double &func, real_1d_array &grad, void *ptr)  {
	// ベクトルのパラメータxを分解して、W、bなどを更新する
	vector<double> vec_x(x.length());
	for (int i = 0; i < x.length(); ++i) {
		vec_x[i] = x[i];
	}
	ae->decodeAndUpdate(vec_x);

	// W、bなどに基づいて、costと偏微分を計算する
	Updates updates = ae->train(lambda, beta, sparsityParam);

    //
    // 関数の値を計算する： f(x0,x1) = 100*(x0+3)^4 + (x1-3)^4 + (x2+1)^2
    // また、偏微分も計算する：df/dx0, df/dx1, df/dx2
    //
    func = updates.cost;
	vector<double> derivatives = ae->encodeDerivatives(updates);
	for (int i = 0; i < derivatives.size(); ++i) {
		grad[i] = derivatives[i];
	}
	/*
    grad[0] = 400*pow(x[0]+3,3);
    grad[1] = 4*pow(x[1]-3,3);
	grad[2] = 2*(x[2]+1);
	*/
}

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
	double pstd = stddev.val[0] * sqrt((X.rows * X.cols) / (double)(X.rows * X.cols - 1)) * 3.0;

	// 各列の値が[-1, 1]の範囲になるようnormalizeする
	X = cv::max(cv::min(X, pstd), -pstd) / pstd;

	// [0.1,0.9]の範囲になるようnormalizeする
	X = (X + 1.0) * 0.4 + 0.1;
}

void test(int numpatches, int patchsize, int hiddenSize) {
	// 以下の行は、最初に１回だけ、小さいデータセットを作成するために必要。
	//MNISTLoader::saveFirstNImages("train-images.idx3-ubyte", 10000, "images10000.idx3-ubyte");

	vector<Mat_<double> > imgs;
	//MNISTLoader::loadImages("images10000.idx3-ubyte", imgs, false);
	loadImages("images.dat", imgs);

	Mat_<double> patches;
	sampleIMAGES(imgs, numpatches, patchsize, patches);

	ae = new AutoEncoder(patches, hiddenSize);

	// BFGSを使って最適化
    real_1d_array x = ae->encodeParams().c_str();//"[0,0,0]";		// 初期値

    double epsg = 0.0000000001;
    double epsf = 0;
    double epsx = 0;
    ae_int_t maxits = 0;
    minlbfgsstate state;
    minlbfgsreport rep;

    minlbfgscreate(1, x, state);
    minlbfgssetcond(state, epsg, epsf, epsx, maxits);
    alglib::minlbfgsoptimize(state, function1_grad);
    minlbfgsresults(state, x, rep);

    printf("termination type: %d\n", int(rep.terminationtype));
    printf("solution: %s\n", x.tostring(2).c_str());


	ae->visualize("weights.png");

	delete ae;
}

int main() {
	test(1000,//10000, // numpatches
		8,		// patchsize
		25		// hiddenSize
		);

	return 0;
}
