#include <iomanip>
#include "BTC.h"

using namespace cv;
using namespace std;

int main(){
	

	// load image 
	Mat src=imread("input.bmp",CV_LOAD_IMAGE_GRAYSCALE);
	Mat dst;

	int BlockSize=8;
	//printf("�п�J�϶��j�p�G");
	//scanf("%d",&BlockSize);

	// keyin param
	int imageQuality;
	if(BlockSize==16)
		imageQuality=35;
	else if (BlockSize==8)
		imageQuality=52;
	else{
		printf("�п�J���檺�v���~��(85~43)�G");
		scanf("%d",&imageQuality);
	}

	printf("ADBTC\n");
	BTC::ADBTC(src,dst,imageQuality);
	cv::imwrite("ADBTC.bmp", dst);

	system("pause");
}