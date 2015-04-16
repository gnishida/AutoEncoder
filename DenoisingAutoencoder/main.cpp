#include "stdafx.h"
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "DenoisingAutoencoder.h"
#include "optimization.h"

/**
 * images.datを読み込み、Denoising Autoencoderにより学習する。
 * inputレイヤからhiddenレイヤへの重みを画像として、weights.pngに保存する。
 * どうも、結果がいまいち。実装が間違っている可能性がある。
 *
 * @author Gen Nishida
 * @date 4/16/2015
 */

using namespace std;
using namespace cv;
using namespace alglib;

DenoisingAutoencoder* ae;

double lambda = 0.0001;
double corruption_level = 0.5;

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
	Updates updates = ae->train(lambda);

    func = updates.cost;
	printf("%d: Cost = %lf\n", count++, updates.cost);

    // 偏微分をgradに格納する
	vector<double> derivatives = ae->encodeDerivatives(updates);
	for (int i = 0; i < derivatives.size(); ++i) {
		grad[i] = derivatives[i];
	}
}

/**
 * 指定されたデータファイルを読み込み、Matの配列を構築する。
 *
 * @param filename		データファイル名
 * @param imgs [OUT]	Matの配列
 */
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
	loadImages("images.dat", imgs);

	Mat_<double> patches;
	sampleIMAGES(imgs, numpatches, patchsize, patches);

	ae = new DenoisingAutoencoder(patches, hiddenSize, corruption_level);

	// BFGSを使って最適化
    real_1d_array x = ae->encodeParams().c_str();	// 初期値

    double epsg = 0;//.0000000001;
    double epsf = 0;
    double epsx = 0;
    ae_int_t maxits = 400;
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
	test(10000, // numpatches
		8,		// patchsize
		25		// hiddenSize
		);

	return 0;
}
