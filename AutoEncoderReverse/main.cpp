#include "stdafx.h"
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "AutoEncoder.h"
#include "optimization.h"
#include "MNISTLoader.h"

/**
 * images10000.idx3-ubyteを読み込み、Sparse Autoencoderにより学習する。
 * inputレイヤからhiddenレイヤへの重みを画像として、weights.pngに保存する。
 *
 * @author Gen Nishida
 * @date 4/16/2015
 */

using namespace std;
using namespace cv;
using namespace alglib;

AutoEncoder* ae;

double lambda = 0.0001;
double beta = 3;
double sparsityParam = 0.01;

/**
 * L-BFGSから呼ばれる関数。
 * コスト関数の値と、勾配ベクトルを返却する。
 *
 * @param x				変数ベクトル
 * @param func [OUT]	コスト関数の値
 * @param grad [OUT]	勾配ベクトル
 * @param ptr
 */
void function1_grad(const real_1d_array &x, double &func, real_1d_array &grad, void *ptr)  {
	static int count = 0;

	// ベクトルのパラメータxを分解して、W、bなどを更新する
	vector<double> vec_x(x.length());
	for (int i = 0; i < x.length(); ++i) {
		vec_x[i] = x[i];
	}
	ae->decodeAndUpdate(vec_x);

	// W、bなどに基づいて、costと偏微分を計算する
	Updates updates = ae->train(lambda, beta, sparsityParam);

    func = updates.cost;
	printf("%d: Cost = %lf\n", count++, updates.cost);

    // 偏微分をgradに格納する
	vector<double> derivatives = ae->encodeDerivatives(updates);
	for (int i = 0; i < derivatives.size(); ++i) {
		grad[i] = derivatives[i];
	}
}

/**
 * 指定された画像リストから、指定された数のパッチを作成する。
 *
 * @param imgs			画像リスト
 * @param num_patches	パッチの数
 * @param patchsize		パッチサイズ（一辺のサイズ）
 * @param X [OUT]		パッチ（各列が1つのパッチを表す。各パッチは、column majorで1列にする。）
 */
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

		// column majorで、パッチを1列にして格納する
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
	vector<Mat_<double> > imgs;
	MNISTLoader::loadImages("images10000.idx3-ubyte", imgs);

	Mat_<double> patches;
	sampleIMAGES(imgs, numpatches, patchsize, patches);
	cout << "Input data is generated." << endl;

	ae = new AutoEncoder(patches, hiddenSize);

	// BFGSを使って最適化
    real_1d_array x = ae->encodeParams().c_str();	// 初期値

    double epsg = 0.0000000001;
    double epsf = 0;
    double epsx = 0;
    ae_int_t maxits = 100;
    minlbfgsstate state;
    minlbfgsreport rep;

    minlbfgscreate(1, x, state);
    minlbfgssetcond(state, epsg, epsf, epsx, maxits);
    alglib::minlbfgsoptimize(state, function1_grad);
    minlbfgsresults(state, x, rep);

	printf("----------------------------------------\n");
	if (rep.terminationtype < 0) {
		printf("Some error occured.\n");
	} else if (rep.terminationtype == 1) {
		printf("Function is converged.\n");
	} else if (rep.terminationtype == 2) {
		printf("Step is converged.\n");
	} else if (rep.terminationtype == 4) {
		printf("Gradient is converged.\n");
	} else if (rep.terminationtype == 5) {
		printf("MaxIts steps was taken.\n");
	} else if (rep.terminationtype == 7) {
		printf("Converged.\n");
	} else {
		printf("Unknown return type: %d\n", rep.terminationtype);
	}

	ae->visualize("weights.png");

	delete ae;
}

int main() {
	test(100,//10000, // numpatches
		28,		// patchsize
		200	// hiddenSize
		);

	return 0;
}
