#include "MNISTLoader.h"

void MNISTLoader::saveFirstNImages(char* inputFileName, int outputNum, char* outputFileName) {
	unsigned char buff[255];
	FILE* fp = fopen(inputFileName, "rb");

	fread(buff, 1, 4, fp);
	int magic_number = BE2LE(buff);

	fread(buff, 1, 4, fp);
	int number_of_images = BE2LE(buff);

	fread(buff, 1, 4, fp);
	int rows = BE2LE(buff);
	fread(buff, 1, 4, fp);
	int cols = BE2LE(buff);

	FILE* fp2 = fopen(outputFileName, "wb");
	LE2BE(magic_number, buff);
	fwrite(buff, 1, 4, fp2);
	LE2BE(outputNum, buff);
	fwrite(buff, 1, 4, fp2);
	LE2BE(rows, buff);
	fwrite(buff, 1, 4, fp2);
	LE2BE(cols, buff);
	fwrite(buff, 1, 4, fp2);

	for (int i = 0; i < outputNum; ++i) {
		// 画像データを読み込む
		Mat_<uchar> img(rows, cols);
		unsigned char* data = new unsigned char[rows * cols];
		fread(data, 1, rows * cols, fp);

		// Matオブジェクトにコピー
		for (int r = 0; r < rows; ++r) {
			for (int c = 0; c < cols; ++c) {
				img(r, c) = data[r * cols + c];
			}
		}
		delete [] data;

		// 画像データをファイルに保存
		for (int r = 0; r < rows; ++r) {
			for (int c = 0; c < cols; ++c) {
				fwrite(&img(r, c), 1, 1, fp2);
			}
		}
	}

	fclose(fp);
	fclose(fp2);
}

void MNISTLoader::loadImages(char* filename, vector<Mat_<uchar> >& imgs, bool convertToBinary) {
	unsigned char buff[255];
	FILE* fp = fopen(filename, "rb");

	int magic_number;
	fread(&magic_number, 4, 1, fp);

	int number_of_images;
	fread(&number_of_images, 4, 1, fp);

	int rows, cols;
	fread(&rows, 4, 1, fp);
	fread(&cols, 4, 1, fp);

	/*fread(buff, 1, 4, fp);
	int magic_number = BE2LE(buff);

	fread(buff, 1, 4, fp);
	int number_of_images = BE2LE(buff);

	fread(buff, 1, 4, fp);
	int rows = BE2LE(buff);
	fread(buff, 1, 4, fp);
	int cols = BE2LE(buff);
	*/
	imgs.resize(number_of_images);

	for (int i = 0; i < number_of_images; ++i) {
		// 画像データを読み込む
		unsigned char* data = new unsigned char[rows * cols];
		fread(data, 1, rows * cols, fp);

		cout << data[0] << endl;
		cout << data[1] << endl;
		cout << data[2] << endl;

		// Matオブジェクトにコピー
		imgs[i] = Mat_<uchar>(rows, cols);
		for (int r = 0; r < rows; ++r) {
			for (int c = 0; c < cols; ++c) {
				if (convertToBinary) {
					imgs[i](r, c) = data[r * cols + c] > 127 ? 1 : 0;
				} else {
					imgs[i](r, c) = data[r * cols + c];
				}
			}
		}
	}
}

int MNISTLoader::BE2LE(unsigned char* buff) {
	return (buff[0] << 24) + (buff[1] << 16) + (buff[2] << 8) + buff[3];
}

void MNISTLoader::LE2BE(int val, unsigned char* buff) {
	buff[0] = val >> 24;
	buff[1] = (val >> 16) & 0xff;
	buff[2] = (val >> 8) & 0xff;
	buff[3] = val & 0xff;
}