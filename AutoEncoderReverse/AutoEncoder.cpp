#include "AutoEncoder.h"
#include <iostream>
#include <sstream>


AutoEncoder::AutoEncoder(const Mat_<double>& data, int hiddenSize) : data(data), hiddenSize(hiddenSize) {
	M = data.cols;
	visibleSize = data.rows;

	double r = sqrt(6.0 / (visibleSize + hiddenSize + 1.0));

	// 重み、バイアスを初期化
	W1 = Mat_<double>(hiddenSize, visibleSize);
	W2 = Mat_<double>(visibleSize, hiddenSize);
	b1 = Mat_<double>::zeros(hiddenSize, 1);
	b2 = Mat_<double>::zeros(visibleSize, 1);

	randu(W1, -r, r);
	randu(W2, -r, r);
}

AutoEncoder::AutoEncoder(char* filename) {
	FILE* fp = fopen(filename, "r");

	int rows, cols;
	fscanf(fp, "%d, %d\n", &rows, &cols);
	W1 = Mat_<double>(rows, cols);
	fscanf(fp, "%d, %d\n", &rows, &cols);
	W2 = Mat_<double>(rows, cols);
	fscanf(fp, "%d\n", &rows);
	b1 = Mat_<double>(rows, 1);
	fscanf(fp, "%d\n", &rows);
	b2 = Mat_<double>(rows, 1);

	visibleSize = W1.cols;
	hiddenSize = W1.rows;

	for (int c = 0; c < W1.cols; ++c) {
		for (int r = 0; r < W1.rows; ++r) {
			fscanf(fp, "%lf\n", &W1(r, c));
		}
	}
	for (int c = 0; c < W2.cols; ++c) {
		for (int r = 0; r < W2.rows; ++r) {
			fscanf(fp, "%lf\n", &W2(r, c));
		}
	}
	for (int r = 0; r < b1.rows; ++r) {
		fscanf(fp, "%lf\n", &b1(r, 0));
	}
	for (int r = 0; r < b2.rows; ++r) {
		fscanf(fp, "%lf\n", &b2(r, 0));
	}

	fclose(fp);
}

/**
 * Autoencoderを１回だけ学習する。
 * 通常、この関数を一定回数実行し、収束させる。
 *
 * @return			コスト
 */
Updates AutoEncoder::train(double lambda, double beta, double sparsityParam) {
	return sparseEncoderCost(W1, W2, b1, b2, lambda, beta, sparsityParam);
}

void AutoEncoder::update(const vector<double>& theta) {
	int index = 0;

	for (int c = 0; c < W1.cols; ++c) {
		for (int r = 0; r < W1.rows; ++r) {
			W1(r, c) = theta[index++];
		}
	}
	for (int c = 0; c < W2.cols; ++c) {
		for (int r = 0; r < W2.rows; ++r) {
			W2(r, c) = theta[index++];
		}
	}
	for (int r = 0; r < b1.rows; ++r) {
		b1(r, 0) = theta[index++];
	}
	for (int r = 0; r < b2.rows; ++r) {
		b2(r, 0) = theta[index++];
	}
}

void AutoEncoder::update(const Updates& updates, double eta) {
	W1 -= eta * updates.dW1;
	W2 -= eta * updates.dW2;
	b1 -= eta * updates.db1;
	b2 -= eta * updates.db2;
}

void AutoEncoder::visualize(char* filename) {
	int n = ceil(sqrt((double)hiddenSize));
	int m = ceil(hiddenSize / (double)n);
	int size = ceil(sqrt((double)visibleSize));

	// 平均を引く
	Mat_<double> X = W1 - mat_avg(W1);

	Mat_<uchar> img = Mat_<uchar>::zeros((size + 1) * m + 1, (size + 1) * n + 1);

	for (int r = 0; r < m; ++r) {
		for (int c = 0; c < n; ++c) {
			int index = r * n + c;
			if (index >= hiddenSize) continue;

			int x0 = (size + 1) * c + 1;
			int y0 = (size + 1) * r + 1;

			// index番目の行を取得する
			Mat_<double> tmp = X.row(index);

			// 絶対値の最大を取得する
			double max_val = mat_max(cv::abs(tmp));

			// 最大値でわる
			tmp = (tmp / max_val + 1) * 127;

			for (int c2 = 0; c2 < size; ++c2) {
				for (int r2 = 0; r2 < size; ++r2) {			
					int index2 = c2 * size + r2;
					if (index2 >= visibleSize) continue;

					img(y0 + r2, x0 + c2) = tmp(0, index2);
				}
			}
		}
	}

	imwrite(filename, img);
}

/**
 * 数値計算により関数fの、xにおける勾配を計算し、返却する。
 *
 * @param func		関数fのポインタ
 * @param x			このポイントにおける勾配を計算する（xは、行ベクトルであること！）
 * @return			勾配ベクトル
 */
Updates AutoEncoder::computeNumericalGradient(double lambda, double beta, double sparsityParam) {
	Updates updates;
	updates.dW1 = Mat_<double>(W1.size());
	updates.dW2 = Mat_<double>(W2.size());
	updates.db1 = Mat_<double>(b1.size());
	updates.db2 = Mat_<double>(b2.size());

	double e = 0.0001;

	for (int r = 0; r < W1.rows; ++r) {
		for (int c = 0; c < W1.cols; ++c) {
			Mat_<double> dW1 = Mat_<double>::zeros(W1.size());
			dW1(r, c) = e;

			Updates u1 = sparseEncoderCost(W1 + dW1, W2, b1, b2, lambda, beta, sparsityParam);
			Updates u2 = sparseEncoderCost(W1 - dW1, W2, b1, b2, lambda, beta, sparsityParam);
			updates.dW1(r, c) = (u1.cost - u2.cost) / e / 2.0;
		}
	}

	for (int r = 0; r < W2.rows; ++r) {
		for (int c = 0; c < W2.cols; ++c) {
			Mat_<double> dW2 = Mat_<double>::zeros(W2.size());
			dW2(r, c) = e;

			Updates u1 = sparseEncoderCost(W1, W2 + dW2, b1, b2, lambda, beta, sparsityParam);
			Updates u2 = sparseEncoderCost(W1, W2 - dW2, b1, b2, lambda, beta, sparsityParam);
			updates.dW2(r, c) = (u1.cost - u2.cost) / e / 2.0;
		}
	}

	for (int r = 0; r < b1.rows; ++r) {
		Mat_<double> db1 = Mat_<double>::zeros(b1.size());
		db1(r, 0) = e;

		Updates u1 = sparseEncoderCost(W1, W2, b1 + db1, b2, lambda, beta, sparsityParam);
		Updates u2 = sparseEncoderCost(W1, W2, b1 - db1, b2, lambda, beta, sparsityParam);
		updates.db1(r, 0) = (u1.cost - u2.cost) / e / 2.0;
	}

	for (int r = 0; r < b2.rows; ++r) {
		Mat_<double> db2 = Mat_<double>::zeros(b2.size());
		db2(r, 0) = e;

		Updates u1 = sparseEncoderCost(W1, W2, b1, b2 + db2, lambda, beta, sparsityParam);
		Updates u2 = sparseEncoderCost(W1, W2, b1, b2 - db2, lambda, beta, sparsityParam);
		updates.db2(r, 0) = (u1.cost - u2.cost) / e / 2.0;
	}

	return updates;
}

/**
 * パラメータを、文字列にシリアライズする。
 *
 * @return	シリアライズされたパラメータ
 */
string AutoEncoder::serializeParams() {
	ostringstream oss;

	oss << "[";
	for (int c = 0; c < W1.cols; ++c) {
		for (int r = 0; r < W1.rows; ++r) {
			oss << W1(r, c) << ",";
		}
	}
	for (int c = 0; c < W2.cols; ++c) {
		for (int r = 0; r < W2.rows; ++r) {
			oss << W2(r, c) << ",";
		}
	}
	for (int r = 0; r < b1.rows; ++r) {
		oss << b1(r, 0) << ",";
	}
	for (int r = 0; r < b2.rows; ++r) {
		oss << b2(r, 0);
		if (r < b2.rows - 1) {
			oss << ",";
		}
	}
	oss << "]";

	return oss.str();
}

/**
 * 偏微分をシリアライズする。
 *
 * @param	
 */
vector<double> AutoEncoder::serializeDerivatives(const Updates& updates) {
	vector<double> ret(W1.rows * W1.cols + W2.rows * W2.cols + b1.rows + b2.rows);
	int index = 0;

	for (int c = 0; c < W1.cols; ++c) {
		for (int r = 0; r < W1.rows; ++r) {
			ret[index++] = updates.dW1(r, c);
		}
	}
	for (int c = 0; c < W2.cols; ++c) {
		for (int r = 0; r < W2.rows; ++r) {
			ret[index++] = updates.dW2(r, c);
		}
	}
	for (int r = 0; r < b1.rows; ++r) {
		ret[index++] = updates.db1(r, 0);
	}
	for (int r = 0; r < b2.rows; ++r) {
		ret[index++] = updates.db2(r, 0);
	}

	return ret;
}

/**
 * 現在のネットワークのパラメータを使って、指定された入力データから、outputを計算し、
 * 指定された数だけ、Top-downによりinputデータを生成する。
 * 元のinputデータ、および、生成されたinputデータを、画像として保存する。
 *
 * @param data			入力データ (dxmの列ベクトル)
 * @param numSamples	生成するinputデータの数
 * @param filename		ファイル名
 */
void AutoEncoder::generateSamples(const Mat_<double>& data, char* filename) {
	int M2 = data.cols;
	int size = ceil(sqrt((double)data.rows));

	Mat_<double> a2(hiddenSize, M2);
	Mat_<double> a3(visibleSize, M2);

	sigmoid(W1 * data + repeat(b1, 1, M2), a2);
	sigmoid(W2 * a2 + repeat(b2, 1, M2), a3);

	// 平均を引く
	Mat_<double> X = data - mat_avg(data);
	Mat_<double> Z = a3 - mat_avg(a3);

	Mat_<uchar> img = Mat_<uchar>::zeros((size + 1) * 2 + 1, (size + 1) * M2 + 1);

	// inputデータ
	for (int m = 0; m < M2; ++m) {
		// 絶対値の最大を取得する
		Mat_<double> tmp1 = X.col(m);
		Mat_<double> tmp2 = Z.col(m);
		double max_val1 = mat_max(cv::abs(tmp1));
		double max_val2 = mat_max(cv::abs(tmp2));

		// 最大値でわる
		tmp1 = (tmp1 / max_val1 + 1) * 127;
		tmp2 = (tmp2 / max_val2 + 1) * 127;

		for (int r = 0; r < size; ++r) {
			for (int c = 0; c < size; ++c) {
				int index = c * size + r;
				img(r + 1, m * (size + 1) + 1 + c) = tmp1(index, 0);
				img(size + 2 + r, m * (size + 1) + 1 + c) = tmp2(index, 0);
			}
		}
	}

	imwrite(filename, img);
}

/**
 * ネットワークのパラメータをファイルに保存する。
 *
 * @param filename	ファイル名
 */
void AutoEncoder::save(char* filename) {
	FILE* fp = fopen(filename, "w");

	fprintf(fp, "%d,%d\n", W1.rows, W1.cols);
	fprintf(fp, "%d,%d\n", W2.rows, W2.cols);
	fprintf(fp, "%d\n", b1.rows);
	fprintf(fp, "%d\n", b2.rows);

	for (int c = 0; c < W1.cols; ++c) {
		for (int r = 0; r < W1.rows; ++r) {
			fprintf(fp, "%lf\n", W1(r, c));
		}
	}
	for (int c = 0; c < W2.cols; ++c) {
		for (int r = 0; r < W2.rows; ++r) {
			fprintf(fp, "%lf\n", W2(r, c));
		}
	}
	for (int r = 0; r < b1.rows; ++r) {
		fprintf(fp, "%lf\n", b1(r, 0));
	}
	for (int r = 0; r < b2.rows; ++r) {
		fprintf(fp, "%lf\n", b2(r, 0));
	}

	fclose(fp);
}

Updates AutoEncoder::sparseEncoderCost(const Mat_<double>& W1, const Mat_<double>& W2, const Mat_<double>& b1, const Mat_<double>& b2, double lambda, double beta, double sparsityParam) {
	Updates updates;
	updates.cost = 0.0f;
	updates.dW1 = Mat_<double>::zeros(hiddenSize, visibleSize);
	updates.dW2 = Mat_<double>::zeros(visibleSize, hiddenSize);
	updates.db1 = Mat_<double>::zeros(hiddenSize, 1);
	updates.db2 = Mat_<double>::zeros(visibleSize, 1);

	// forward pass
	Mat_<double> rho_hat = Mat_<double>::zeros(hiddenSize, 1);
	Mat_<double> a2(hiddenSize, M);
	Mat_<double> a3(visibleSize, M);

	sigmoid(W1 * data + repeat(b1, 1, M), a2);
	sigmoid(W2 * a2 + repeat(b2, 1, M), a3);
	reduce(a2, rho_hat, 1, CV_REDUCE_AVG);
	updates.cost = mat_sum((a3 - data).mul(a3 - data)) * 0.5 / M + lambda * 0.5 * (mat_sum(W1.mul(W1)) + mat_sum(W2.mul(W2)));

	// back propagation
	Mat_<double> delta3 = -(data - a3).mul(a3).mul(1 - a3);
	Mat_<double> delta2 = (W2.t() * delta3 + beta * repeat(-sparsityParam / rho_hat + (1-sparsityParam) / (1-rho_hat), 1, M)).mul(a2).mul(1 - a2);
	updates.dW1 = delta2 * data.t() / M + lambda * W1;
	updates.dW2 = delta3 * a2.t() / M + lambda * W2;
	reduce(delta2, updates.db1, 1, CV_REDUCE_AVG);
	reduce(delta3, updates.db2, 1, CV_REDUCE_AVG);

	// sparsity penalty
	Mat log1, log2;
	cv::log(sparsityParam / rho_hat, log1);
	//log1 /= exp(1.0);
	cv::log((1-sparsityParam) / (1 - rho_hat), log2);
	//log2 /= exp(1.0);

	updates.cost += beta * mat_sum(sparsityParam * log1 + (1-sparsityParam) * log2);

	return updates;
}

/**
 * 行列の各要素について、sigmoid関数を適用した結果を返却する。
 *
 * @param z		元の行列
 * @return		計算結果の行列
 */
Mat_<double> AutoEncoder::sigmoid(const Mat_<double>& z) {
	Mat_<double> ret(z.size());

	for (int r = 0; r < z.rows; ++r) {
		for (int c = 0; c < z.cols; ++c) {
			ret(r, c) = 1.0 / (1.0 + exp(-z(r, c)));
		}
	}

	return ret;
}

void AutoEncoder::sigmoid(const Mat_<double>& z, Mat_<double>& ret) {
	ret = Mat_<double>(z.size());

	for (int r = 0; r < z.rows; ++r) {
		for (int c = 0; c < z.cols; ++c) {
			ret(r, c) = 1.0 / (1.0 + exp(-z(r, c)));
		}
	}
}

/**
 * 行列の要素の合計を返却する。
 *
 * @param m		行列
 * @return		要素の合計
 */
double AutoEncoder::mat_sum(const Mat_<double>& m) {
	Mat_<double> tmp;
	reduce(m, tmp, 0, CV_REDUCE_SUM);
	reduce(tmp, tmp, 1, CV_REDUCE_SUM);
	return tmp(0, 0);
}

/**
 * 行列の要素の平均を返却する。
 *
 * @param m		行列
 * @return		要素の合計
 */
double AutoEncoder::mat_avg(const Mat_<double>& m) {
	Mat_<double> tmp;
	reduce(m, tmp, 0, CV_REDUCE_AVG);
	reduce(tmp, tmp, 1, CV_REDUCE_AVG);
	return tmp(0, 0);
}

/**
 * 行列の要素の最大値を返却する。
 *
 * @param m		行列
 * @return		要素の合計
 */
double AutoEncoder::mat_max(const Mat_<double>& m) {
	Mat_<double> tmp;
	reduce(m, tmp, 0, CV_REDUCE_MAX);
	reduce(tmp, tmp, 1, CV_REDUCE_MAX);
	return tmp(0, 0);
}

/**
 * 行列の要素の最小値を返却する。
 *
 * @param m		行列
 * @return		要素の合計
 */
double AutoEncoder::mat_min(const Mat_<double>& m) {
	Mat_<double> tmp;
	reduce(m, tmp, 0, CV_REDUCE_MIN);
	reduce(tmp, tmp, 1, CV_REDUCE_MIN);
	return tmp(0, 0);
}