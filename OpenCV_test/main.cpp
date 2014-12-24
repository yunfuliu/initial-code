#include <iomanip>
#include "BTC.h"

using namespace cv;
using namespace std;

int main(){
	

	// load image 
	Mat src=imread("input.bmp",CV_LOAD_IMAGE_GRAYSCALE);
	Mat dst;

	int BlockSize=8;
	//printf("請輸入區塊大小：");
	//scanf("%d",&BlockSize);

	// keyin param
	int imageQuality;
	if(BlockSize==16)
		imageQuality=35;
	else if (BlockSize==8)
		imageQuality=52;
	else{
		printf("請輸入期望的影像品質(85~43)：");
		scanf("%d",&imageQuality);
	}

	printf("ADBTC\n");
	BTC::ADBTC(src,dst,imageQuality);
	cv::imwrite("ADBTC.bmp", dst);

	system("pause");
}